#include "sparseBlockedAttention.h"

// sparseBlockedGEMM wrapper
// mat1 -> (H, (m1*k1 + m2*k2 + ...)), mat2 -> (H, (n1*k1 + n2*k2 + ...)), out -> (H, (m1*n1 + m2*n2 + ...))
// accum1 -> (0, m1*k1, (m1*k1) + m2*k2...), accum2 -> (0, n1*k1, (n1*k1) + n2*k2...), accumOut -> (0, m1*n1, (m1*n1) + m2*n2...)
// MList -> (m1, m2, ...), NList -> (n1, n2, ...), KList -> (k1, k2, ...)
// default transpose1 = false: mat1(M, K) -> true: mat1(K, M)
// default transpose2 = false: mat2(N, K) -> true: mat1(K, N)
class sparseBlockedGEMM: public torch::autograd::Function<sparseBlockedGEMM>{
    public:
    static variableList forward(
        autogradContext *context, const variable &mat1, const variable &mat2,
        const torch::optional<variable> &mat1Accum, const torch::optional<variable> &mat2Accum, const torch::optional<variable> &outAccum,
        const torch::optional<variable> &MList, const torch::optional<variable> &NList, const torch::optional<variable> &KList,
        const bool transpose1, const bool transpose2
    ){
        assert((mat1Accum.has_value() && mat2Accum.has_value() && outAccum.has_value()) || (MList.has_value() && NList.has_value() && KList.has_value()));

        torch::Tensor mat1AccumVal, mat2AccumVal, outAccumVal;
        torch::Tensor MListVal, NListVal, KListVal;

        if (!mat1Accum.has_value()){
            MListVal = MList.value(); NListVal = NList.value(); KListVal = KList.value();

            torch::Tensor __zero = torch::zeros({1}, MListVal.options());
            mat1AccumVal = torch::cat({__zero, (MListVal * KListVal)}, 0).cumsum(0);
            mat2AccumVal = torch::cat({__zero, (NListVal * KListVal)}, 0).cumsum(0);
            outAccumVal = torch::cat({__zero, (MListVal * NListVal)}, 0).cumsum(0);
        }
        else if (!MList.has_value()){
            mat1AccumVal = mat1Accum.value(); mat2AccumVal = mat2Accum.value(); outAccumVal = outAccum.value();

            auto mk = mat1AccumVal.index({torch::indexing::Slice(1)}) - mat1AccumVal.index({torch::indexing::Slice(torch::indexing::None, -1)});
            auto nk = mat2AccumVal.index({torch::indexing::Slice(1)}) - mat2AccumVal.index({torch::indexing::Slice(torch::indexing::None, -1)});
            auto mn = outAccumVal.index({torch::indexing::Slice(1)}) - outAccumVal.index({torch::indexing::Slice(torch::indexing::None, -1)});

            MListVal = ((mk * mn) / nk).sqrt().to(torch::kInt64);
            NListVal = (mn / MListVal).to(torch::kInt64);
            KListVal = (mk / MListVal).to(torch::kInt64);
        }
        else{
            mat1AccumVal = mat1Accum.value(); mat2AccumVal = mat2Accum.value(); outAccumVal = outAccum.value();
            MListVal = MList.value(); NListVal = NList.value(); KListVal = KList.value();
        }
        context->save_for_backward({mat1, mat2, mat1AccumVal, mat2AccumVal, outAccumVal, MListVal, NListVal, KListVal});
        context->saved_data["transpose1"] = transpose1;
        context->saved_data["transpose2"] = transpose2;
        auto out = sparseBlockedGEMMLaunch(mat1, mat2, mat1AccumVal, mat2AccumVal, outAccumVal, MListVal, NListVal, KListVal, transpose1, transpose2);
        return {out};
    }

    static variableList backward(autogradContext *context, const variableList &grads){
        auto outGrad = grads[0];
        auto fwdSaved = context->get_saved_variables();

        auto mat1 = fwdSaved[0].to(outGrad.scalar_type());
        auto mat2 = fwdSaved[1].to(outGrad.scalar_type());
        auto mat1Accum = fwdSaved[2];
        auto mat2Accum = fwdSaved[3];
        auto outAccum = fwdSaved[4];
        auto MList = fwdSaved[5];
        auto NList = fwdSaved[6];
        auto KList = fwdSaved[7];

        auto transpose1 = context->saved_data["transpose1"].toBool();
        auto transpose2 = context->saved_data["transpose2"].toBool();

        auto mat1Grad = sparseBlockedGEMMLaunch(outGrad, mat2, outAccum, mat2Accum, mat1Accum, MList, KList, NList, false, !transpose2);
        auto mat2Grad = sparseBlockedGEMMLaunch(outGrad, mat1, outAccum, mat1Accum, mat2Accum, NList, KList, MList, true, !transpose1);

        return {mat1Grad, mat2Grad, variable(), variable(), variable(), variable(), variable(), variable(), variable(), variable()};
    }
};

// sparseBlockedSoftmax wrapper (only for last dim)
// src -> (H, (m1*k1 + m2*k2 + ...)), out -> (H, (m1*k1 + m2*k2 + ...))
// accumSrc -> (0, m1*k1, (m1*k1) + m2*k2...)
// rowList -> (m1, m2, ...), colList -> (k1, k2, ...)
class sparseBlockedSoftmax: public torch::autograd::Function<sparseBlockedSoftmax>{
    public:
    static variableList forward(autogradContext *context, const variable &src, const torch::optional<variable> &rowList, const torch::optional<variable> &colList, const torch::optional<variable> &accumSrc){
        assert(rowList.has_value() && colList.has_value());

        torch::Tensor accumVal, rowVal, colVal;
        if (!(accumSrc.has_value())){
            rowVal = rowList.value();
            colVal = colList.value();

            torch::Tensor __zero = torch::zeros({1}, rowVal.options());
            accumVal = torch::cat({__zero, rowVal * colVal}, 0).cumsum(0);
        }
        else{
            accumVal = accumSrc.value();
            rowVal = rowList.value();
            colVal = colList.value();
        }

        auto out = sparseBlockedSoftmaxLaunch(src, accumVal, rowVal, colVal);
        context->save_for_backward({out, accumVal, rowVal, colVal});
        return {out};
    }

    static variableList backward(autogradContext *context, const variableList &grads){
        auto outGrad = grads[0];
        auto fwdSaved = context->get_saved_variables();

        auto out = fwdSaved[0].to(outGrad.scalar_type());
        auto accumSrc = fwdSaved[1];
        auto rowList = fwdSaved[2];
        auto colList = fwdSaved[3];

        auto srcGrad = sparseBlockedSoftmaxBackwardLaunch(out, outGrad, accumSrc, rowList, colList);
        return {srcGrad, variable(), variable(), variable()};
    }
};


torch::Tensor sparseBlockedGEMM_op(
    const variable &mat1, const variable &mat2,
    const torch::optional<variable> &mat1Accum, const torch::optional<variable> &mat2Accum, const torch::optional<variable> &outAccum,
    const torch::optional<variable> &MList, const torch::optional<variable> &NList, const torch::optional<variable> &KList,
    const bool transpose1, const bool transpose2
){
    return sparseBlockedGEMM::apply(mat1, mat2, mat1Accum, mat2Accum, outAccum, MList, NList, KList, transpose1, transpose2)[0];
}

torch::Tensor sparseBlockedSoftmax_op(
    const variable &src, const torch::optional<variable> &rowList, const torch::optional<variable> &colList, const torch::optional<variable> &srcAccum
){
    return sparseBlockedSoftmax::apply(src, rowList, colList, srcAccum)[0];
}

#if BUILD_PYBIND11 == 1
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
        m.def("sparseBlockedGEMM", &sparseBlockedGEMM_op, "sparseBlockedGEMM forward");
        m.def("sparseBlockedSoftmax", &sparseBlockedSoftmax_op, "sparseBlockedSoftmax forward");
    };
#endif

int main(){
    auto torchType = torch::kHalf;

    // auto dummyC = torch::randn({5, 8, 60, 32}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA));
    // auto dummyCT = torch::randn({5, 8, 32, 60}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA));
    // auto checkRes = torch::matmul(dummyC, dummyCT);

    // auto dummyA = torch::randn({8, 100, 32}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA));
    // auto dummyB = torch::randn({8, 100, 32}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA));

    // auto graphLength = torch::tensor({0, 10, 5, 20, 5, 60}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto powAccumLength = graphLength.pow(2).cumsum(0);
    // auto powGraphLength = (graphLength * 32).cumsum(0);

    // auto context = autogradContext();
    // auto res = sparseQKAttention::forward(&context, dummyA, dummyB, powGraphLength, powAccumLength)[0];

    // auto dummyA = torch::randn(2*7*10, c10::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({2, 7, 10});
    // auto dummyB = torch::randn(2*7*10, c10::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({2, 7, 10});
    // auto graphLength = torch::tensor({0, 7}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto accumLength = (graphLength * 10).cumsum(0);
    // auto powAccumLength = graphLength.pow(2).cumsum(0);
    // auto context = autogradContext();

    // auto res = sparseQKAttention::forward(&context, dummyA, dummyB, accumLength, powAccumLength)[0];
    // auto checkRes = torch::matmul(dummyA, dummyB.transpose(-1, -2));
    // std::cout << res.reshape({2, 7, 7}) << std::endl;
    // std::cout << checkRes << std::endl;
    // int i = 0;

    // const int m1 = 33;
    // const int n1 = 33;
    // const int k1 = 61;

    // const int m2 = 127;
    // const int n2 = 127;
    // const int k2 = 61;

    // const int h = 128;

    // auto dummyC = torch::randn({h, 2, 127, 127}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).requires_grad_();
    // auto dummyD = torch::randn({h, 2, 127, 127}, c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).requires_grad_();
    // auto res = dummyC.softmax(-1);
    // auto torchRes = torch::matmul(dummyC, dummyD);

    // auto dummyA = torch::randn(h*(m1*k1 + m2*k2), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();
    // auto dummyB = torch::randn(h*(n1*k1 + n2*k2), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();

    // auto mat1Accum = torch::tensor({0, m1*k1, m1*k1+m2*k2}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto mat2Accum = torch::tensor({0, n1*k1, n1*k1+n2*k2}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto outAccum = torch::tensor({0, m1*n1, m1*n1+m2*n2}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // auto res = sparseBlockedGEMM::apply(dummyA, dummyB, mat1Accum, mat2Accum, outAccum, torch::optional<variable>(), torch::optional<variable>(), torch::optional<variable>(), false, false)[0];

    // torchRes.mean().backward();
    // res.mean().backward();

    // auto checkRes = torch::cat({torch::matmul(dummyA.index({"...", torch::indexing::Slice(torch::indexing::None, m1*k1)}).reshape({-1, m1, k1}), dummyB.index({"...", torch::indexing::Slice(torch::indexing::None, n1*k1)}).reshape({-1, n1, k1}).transpose(-1, -2)).reshape({h, -1}), torch::matmul(dummyA.index({"...", torch::indexing::Slice(m1*k1)}).reshape({-1, m2, k2}), dummyB.index({"...", torch::indexing::Slice(n1*k1)}).reshape({-1, n2, k2}).transpose(-1, -2)).reshape({h, -1})}, -1);

    // std::cout << (res - checkRes).abs().mean() << std::endl;

    

    // std::cout << checkRes[0].index({torch::indexing::Slice(torch::indexing::None, g1*g1)}).reshape({g1, g1}) << std::endl;
    // std::cout << res[0].index({torch::indexing::Slice(torch::indexing::None, g1*g1)}).reshape({g1, g1}) << std::endl;
    // auto sli1 = checkRes[1].index({torch::indexing::Slice(25+120*130, 25+125*130)}).reshape({5,130});
    // auto sli2 = res[1].index({torch::indexing::Slice(25+120*130, 25+125*130)}).reshape({5,130});

    // std::cout << sli1.ne(sli2).nonzero() << std::endl;
    // std::cout << checkRes[0].index({torch::indexing::Slice(25+120*130, 25+125*130)}).reshape({5,130}) << std::endl;
    // std::cout << res[0].index({torch::indexing::Slice(25+120*130, 25+125*130)}).reshape({5,130}) << std::endl;

    // std::cout << res[0].index({torch::indexing::Slice(25, 25+260)}).reshape({2, 130}) << std::endl;
    // std::cout << res.eq(checkRes).all() << std::endl;
    // std::cout << res.eq(checkRes).sum(-1) << std::endl;

    // int m = 11;
    // int n = 13;
    // int k = 5;
    // int h = 1;

    // auto dummyA = torch::arange(h*(m * k), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();
    // auto dummyB = torch::arange(h*(n * k), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();

    // auto dummyC = torch::arange(h*(m * k), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();
    // auto dummyD = torch::arange(h*(n * k), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1}).requires_grad_();

    // dummyA.retain_grad();
    // dummyB.retain_grad();
    // dummyC.retain_grad();
    // dummyD.retain_grad();

    // auto mat1Accum = torch::tensor({0, m*k}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto mat2Accum = torch::tensor({0, n*k}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto outAccum = torch::tensor({0, m*n}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // auto MList = torch::tensor({m}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto NList = torch::tensor({n}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    // auto KList = torch::tensor({k}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // auto checkRes = torch::matmul(dummyC.reshape({h, m, k}), dummyD.reshape({h, n, k}).transpose(-1, -2));
    // auto res = sparseBlockedGEMM::apply(dummyA, dummyB, torch::optional<variable>(), torch::optional<variable>(), torch::optional<variable>(), MList, NList, KList, false, false)[0];

    // checkRes.retain_grad();
    // res.retain_grad();

    // auto dummyGrad = torch::full({h, m*n}, 1, c10::TensorOptions().dtype(torchType).device(torch::kCUDA));

    // checkRes.backward(dummyGrad.reshape({h, m, n}));

    // res.backward(dummyGrad);

    // std::cout << (res - checkRes.reshape({h, -1})).abs().max() << std::endl;
    // std::cout << (res.grad() - checkRes.grad().reshape({h, -1})).abs().max() << std::endl;
    // std::cout << (dummyA.grad() - dummyC.grad()).abs().max() << std::endl;
    // std::cout << (dummyB.grad() - dummyD.grad()).abs().max() << std::endl;

    // // std::cout << (res.reshape({8, l, l}) - checkRes).abs().mean() << std::endl;
    // std::cout << dummyC.grad() << std::endl;
    // std::cout << dummyA.grad() << std::endl;
    // std::cout << dummyD.grad() << std::endl;
    // std::cout << dummyB.grad() << std::endl;

    // uint i = 0;

    const int m = 127;
    const int n = 2047;
    const int h = 7;

    auto dummyA = torch::randn(h*(3*m*n), c10::TensorOptions().dtype(torchType).device(torch::kCUDA)).reshape({h, -1});
    auto dummyB = torch::zeros_like(dummyA, dummyA.options());
    dummyB.copy_(dummyA);

    dummyA.requires_grad_();
    dummyA.retain_grad();
    dummyB.requires_grad_();
    dummyB.retain_grad();

    dummyB = dummyB.reshape({h, 3, m, n});

    auto rowList = torch::tensor({m, m, m}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    auto colList = torch::tensor({n, n, n}, c10::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    auto checkRes = dummyB.softmax(-1);
    auto res = sparseBlockedSoftmax::apply(dummyA, rowList, colList, torch::optional<variable>())[0];

    checkRes.retain_grad();
    res.retain_grad();

    checkRes.mean().backward();
    res.mean().backward();

    std::cout << res.reshape({h, 3, m, n})[1][1][1] << std::endl;
    std::cout << checkRes[1][1][1] << std::endl;
    std::cout << (res - checkRes.reshape({h, -1})).abs().max() << std::endl;
    // std::cout << (dummyA.grad() - dummyB.grad().reshape({h, -1})).abs().max() << std::endl;
}
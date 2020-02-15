//
// Created by Michael Curry on 2/15/20.
//

#ifndef TORCH_TEST_NET_H
#define TORCH_TEST_NET_H

#include "doctest.h"
struct Net : torch::nn::Module {
    Net():
            fc1(register_module("fc1", torch::nn::Linear(784, 64))),
            fc2(register_module("fc2", torch::nn::Linear(64, 64))),
            fc3(register_module("fc3", torch::nn::Linear(64, 10)))
    {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x.view({-1, 784})));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc2(x));
        x = torch::log_softmax(fc3(x), /*dim=*/1);
        return x;
    }

    torch::Tensor operator()(torch::Tensor x) {
        return forward(x);
    }

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

DOCTEST_TEST_CASE("doctest test case") {
    Net net = Net();
    DOCTEST_SUBCASE("forward pass has right shape") {
        torch::Tensor x1 = torch::rand({2,784});
        torch::Tensor result = net(x1);
        DOCTEST_CHECK(result.sizes()[0] == 2);
        DOCTEST_CHECK(result.sizes()[1] == 10);
    }
}
#endif //TORCH_TEST_NET_H


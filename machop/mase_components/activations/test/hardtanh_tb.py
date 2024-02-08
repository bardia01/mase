import torch

from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb

from cocotb.triggers import Timer

import pytest
import cocotb

from mase_cocotb.runner import mase_runner

pytestmark = pytest.mark.simulator_required


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply


# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    bitwidth = 8
    bias = 1
    num = 4
    max_val = 4
    min_val = -4
    def __init__(self, samples=10):
        self.m = nn.Hardtanh(min_val = self.min_val, max_val = self.max_val)
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = torch.rand(self.num)
        r1, r2 = 100, -100
        xs = (r1 - r2) * xs + r2
        # 8-bit, (5, 3)
        xs = quantize(xs, self.bitwidth, self.bias)
        return xs, self.m(xs)

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_PRECISION_0" : 8,
            "DATA_IN_0_TENSOR_SIZE_DIM_0": 1,
            "DATA_IN_0_PARALLELISM_DIM_0": 2,
            "DATA_IN_0_PARALLELISM_DIM_1": 2,          
            "DATA_OUT_0_PRECISION_0": 8,
            "DATA_OUT_0_PARALLELISM_DIM_0": 2,
            "DATA_OUT_0_PARALLELISM_DIM_1": 2,
            "MIN_VAL": self.min_val * 2**self.bias,
            "MAX_VAL": self.max_val * 2**self.bias,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        print("Inputs: ", inputs)
        shifted_integers = (inputs * (2**self.bias)).int()
        print("Shifted integers input: ", shifted_integers)
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        print("Outputs: ", outputs)
        shifted_integers = (outputs * (2**self.bias)).int()
        print("Shifted integers output: ", shifted_integers)
        return shifted_integers.numpy().tolist()


def int_list_to_bin_list(l):
    a = []
    for i in l:
        a.append(i.signed_integer)
    return a

@cocotb.test()
async def test_hardtanh(dut):
    """Test integer based Relu"""
    test_case = VerificationCase(samples=1)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        await Timer(2, units="ns")
        dut_out = int_list_to_bin_list(dut.data_out_0.value)
        assert dut_out == y, f"output q was incorrect on the {i}th cycle\ninput was {x}"


if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])

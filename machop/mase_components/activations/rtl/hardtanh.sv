module hardtanh #(
    /* verilator lint_off UNUSEDPARAM */
    parameter MAX_VAL = 127,
    parameter MIN_VAL = -128,

    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0
)(
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready,
);
  initial begin
    $display("BARDAIIARIADIA - loop size = %d",DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1);
  end

  parameter loop_size = DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1;

  for (genvar i = 0; i < loop_size; i++) begin : Hardtanh
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in_0[i]) <= $signed(MIN_VAL)) data_out_0[i] = DATA_OUT_0_PRECISION_0'(MIN_VAL);
      else if ($signed(data_in_0[i]) >= $signed(MAX_VAL)) data_out_0[i] = DATA_OUT_0_PRECISION_0'(MAX_VAL);
      else data_out_0[i] = data_in_0[i];

    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule

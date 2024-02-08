module tb();

parameter MAX_VAL = 127,
parameter MIN_VAL = -128,

parameter DATA_IN_0_PRECISION_0 = 8,
parameter DATA_IN_0_PRECISION_1 = 1,
parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

parameter DATA_OUT_0_PRECISION_0 = 8,
parameter DATA_OUT_0_PRECISION_1 = 0,
parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 0,
parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 0,
parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
parameter DATA_OUT_0_PARALLELISM_DIM_1 = 0,

parameter INPLACE = 0


function logic [DATA_OUT_0_PRECISION_0-1:0] model_hardtanh [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0](
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0
);
    if ($signed(data_in_0) <= MIN_VAL) model_hardtanh = 'MIN_VAL;
    else if ($signed(data_in_0) >= MAX_VAL) model_hardtanh = 'MAX_VAL;
    else data_out_0 = model_hardtanh;
endfunction

hardtanh dut (
    .rst(1'b0),
    .clk(1'b0),
    .data_in_0_valid(1'b0),
    .data_out_0_ready(1'b0)
);

initial begin
    logic [DATA_IN_0_PRECISION_0-1:0] data_in_0;
    
    for(int i = 0; i < 2**DATA_IN_0_PRECISION_0; i++) begin
        assert(model_hardtanh(i) == );
        $display("data_in_0 = %d, data_out_0 = %d", data_in_0, model_hardtanh(data_in_0));
    end
end
endmodule
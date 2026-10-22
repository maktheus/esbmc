// Lema: td(v) (usado pelo gerador) == v / 256 (divisao do C: trunca para zero), para TODO v de 32 bits.
module lema_td(input signed [31:0] v, output ok, output signed [31:0] a, b);
  function signed [31:0] td; input signed [31:0] v;
    td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction
  assign a = td(v);
  assign b = v / 32'sd256;
  assign ok = (a == b);
endmodule

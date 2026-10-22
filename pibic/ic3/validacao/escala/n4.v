// Gerado por gen_transition_system.py — NAO EDITAR A MAO
// Actor DDPG real: 4 -> 4 -> 4 -> 1, Q8.8, 45 parametros
// Estado: 16 bits por variavel. Propriedade: |th| <= 53 (12 graus)

module top(input clk,
           input signed [15:0] i_x, i_xd, i_th, i_thd,
           output bad);

  reg started = 0;
  reg signed [15:0] x = 0, xd = 0, th = 0, thd = 0;

  wire signed [31:0] X   = {{16{x[15]}},x};
  wire signed [31:0] XD  = {{16{xd[15]}},xd};
  wire signed [31:0] TH  = {{16{th[15]}},th};
  wire signed [31:0] THD = {{16{thd[15]}},thd};

  // divisao inteira por 256 truncando para ZERO — identica ao / do C.
  // NAO usar >>> 8 sozinho: arredonda para -infinito e divergiria
  // do harness verificado para valores negativos.
  function signed [31:0] td; input signed [31:0] v;
    td = v[31] ? -((-v) >>> 8) : (v >>> 8); endfunction

  // clamp em [-5,5] — sobrejetor sobre a regiao inicial exata
  function signed [15:0] ci; input signed [15:0] v;
    ci = (v >  16'sd5) ?  16'sd5 :
         (v < -16'sd5) ? -16'sd5 : v; endfunction

  // camada 1: divisao POR TERMO, como verify_ddpg_closed_loop.py:43-44
  wire signed [31:0] pre1_0 = td(32'sd31*X) + td((-32'sd146)*XD) + td(32'sd104*TH) + td((-32'sd251)*THD) + (-32'sd47);
  wire signed [31:0] h1_0 = pre1_0[31] ? 32'sd0 : pre1_0;
  wire signed [31:0] pre1_1 = td((-32'sd226)*X) + td(32'sd248*XD) + td((-32'sd204)*TH) + td(32'sd74*THD) + (-32'sd3);
  wire signed [31:0] h1_1 = pre1_1[31] ? 32'sd0 : pre1_1;
  wire signed [31:0] pre1_2 = td(32'sd296*X) + td((-32'sd241)*XD) + td(32'sd219*TH) + td((-32'sd81)*THD) + (-32'sd41);
  wire signed [31:0] h1_2 = pre1_2[31] ? 32'sd0 : pre1_2;
  wire signed [31:0] pre1_3 = td((-32'sd262)*X) + td((-32'sd212)*XD) + td(32'sd144*TH) + td(32'sd128*THD) + 32'sd44;
  wire signed [31:0] h1_3 = pre1_3[31] ? 32'sd0 : pre1_3;

  // camada 2
  wire signed [31:0] pre2_0 = td((-32'sd170)*h1_0) + td(32'sd89*h1_1) + td((-32'sd137)*h1_2) + td((-32'sd86)*h1_3) + 32'sd10;
  wire signed [31:0] h2_0 = pre2_0[31] ? 32'sd0 : pre2_0;
  wire signed [31:0] pre2_1 = td(32'sd122*h1_0) + td(32'sd121*h1_1) + td(32'sd98*h1_2) + td((-32'sd169)*h1_3) + 32'sd43;
  wire signed [31:0] h2_1 = pre2_1[31] ? 32'sd0 : pre2_1;
  wire signed [31:0] pre2_2 = td(32'sd95*h1_0) + td(32'sd99*h1_1) + td(32'sd3*h1_2) + td((-32'sd175)*h1_3) + (-32'sd28);
  wire signed [31:0] h2_2 = pre2_2[31] ? 32'sd0 : pre2_2;
  wire signed [31:0] pre2_3 = td((-32'sd87)*h1_0) + td((-32'sd177)*h1_1) + td(32'sd85*h1_2) + td((-32'sd132)*h1_3) + (-32'sd34);
  wire signed [31:0] h2_3 = pre2_3[31] ? 32'sd0 : pre2_3;

  // saida
  wire signed [31:0] z = td(32'sd92*h2_0) + td((-32'sd43)*h2_1) + td(32'sd86*h2_2) + td(32'sd149*h2_3) + (-32'sd18);

  // aproximacao linear de tanh em Q8.8 — mesma de TANH_APPROX_C
  wire signed [31:0] z_abs = z[31] ? -z : z;
  wire signed [31:0] tanh_abs =
      (z_abs <=  32'sd64) ? td(z_abs * 32'sd252) :
      (z_abs <= 32'sd192) ?  32'sd62 + td((z_abs -  32'sd64) * 32'sd200) :
      (z_abs <= 32'sd384) ? 32'sd162 + td((z_abs - 32'sd192) *  32'sd92) :
      (z_abs <= 32'sd768) ? 32'sd231 + td((z_abs - 32'sd384) *  32'sd16) :
                            32'sd255;
  wire signed [31:0] tanh_z = z[31] ? -tanh_abs : tanh_abs;
  wire signed [31:0] F_Q = tanh_z * 32'sd10;   // (t*10*256)/256 == t*10

  // dinamica linearizada Q8.8 — mesma de verify_ddpg_closed_loop.py:185-187
  wire signed [31:0] th_acc = td(32'sd4040*TH - 32'sd375*F_Q);
  wire signed [31:0] n_x   = X   + td(32'sd5*XD);
  wire signed [31:0] n_xd  = XD  + td(32'sd5*F_Q);
  wire signed [31:0] n_th  = TH  + td(32'sd5*THD);
  wire signed [31:0] n_thd = THD + td(32'sd5*th_acc);

  always @(posedge clk) begin
    started <= 1;
    if (!started) begin
      x <= ci(i_x); xd <= ci(i_xd); th <= ci(i_th); thd <= ci(i_thd);
    end else begin
      x <= n_x[15:0];   xd  <= n_xd[15:0];
      th <= n_th[15:0]; thd <= n_thd[15:0];
    end
  end

  // propriedade de seguranca: o pendulo nunca passa de 12 graus
  assign bad = (th > 16'sd53) || (th < -16'sd53);
endmodule

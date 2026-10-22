// Gerado por gen_transition_system.py — NAO EDITAR A MAO
// Actor DDPG real: 4 -> 12 -> 12 -> 1, Q8.8, 229 parametros
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
  wire signed [31:0] pre1_0 = td(32'sd31*X) + td((-32'sd146)*XD) + td(32'sd104*TH) + td((-32'sd251)*THD) + 32'sd31;
  wire signed [31:0] h1_0 = pre1_0[31] ? 32'sd0 : pre1_0;
  wire signed [31:0] pre1_1 = td((-32'sd226)*X) + td(32'sd248*XD) + td((-32'sd204)*TH) + td(32'sd74*THD) + (-32'sd40);
  wire signed [31:0] h1_1 = pre1_1[31] ? 32'sd0 : pre1_1;
  wire signed [31:0] pre1_2 = td(32'sd296*X) + td((-32'sd241)*XD) + td(32'sd219*TH) + td((-32'sd81)*THD) + (-32'sd48);
  wire signed [31:0] h1_2 = pre1_2[31] ? 32'sd0 : pre1_2;
  wire signed [31:0] pre1_3 = td((-32'sd262)*X) + td((-32'sd212)*XD) + td(32'sd144*TH) + td(32'sd128*THD) + (-32'sd49);
  wire signed [31:0] h1_3 = pre1_3[31] ? 32'sd0 : pre1_3;
  wire signed [31:0] pre1_4 = td((-32'sd229)*X) + td((-32'sd54)*XD) + td((-32'sd208)*TH) + td(32'sd264*THD) + (-32'sd12);
  wire signed [31:0] h1_4 = pre1_4[31] ? 32'sd0 : pre1_4;
  wire signed [31:0] pre1_5 = td(32'sd134*X) + td((-32'sd240)*XD) + td(32'sd279*TH) + td((-32'sd174)*THD) + 32'sd63;
  wire signed [31:0] h1_5 = pre1_5[31] ? 32'sd0 : pre1_5;
  wire signed [31:0] pre1_6 = td((-32'sd72)*X) + td(32'sd296*XD) + td((-32'sd237)*TH) + td(32'sd290*THD) + 32'sd45;
  wire signed [31:0] h1_6 = pre1_6[31] ? 32'sd0 : pre1_6;
  wire signed [31:0] pre1_7 = td(32'sd299*X) + td(32'sd106*XD) + td((-32'sd250)*TH) + td((-32'sd74)*THD) + 32'sd16;
  wire signed [31:0] h1_7 = pre1_7[31] ? 32'sd0 : pre1_7;
  wire signed [31:0] pre1_8 = td((-32'sd253)*X) + td(32'sd270*XD) + td((-32'sd164)*TH) + td((-32'sd4)*THD) + 32'sd55;
  wire signed [31:0] h1_8 = pre1_8[31] ? 32'sd0 : pre1_8;
  wire signed [31:0] pre1_9 = td(32'sd129*X) + td((-32'sd153)*XD) + td(32'sd253*TH) + td((-32'sd180)*THD) + 32'sd52;
  wire signed [31:0] h1_9 = pre1_9[31] ? 32'sd0 : pre1_9;
  wire signed [31:0] pre1_10 = td(32'sd284*X) + td(32'sd15*XD) + td(32'sd273*TH) + td((-32'sd115)*THD) + 32'sd28;
  wire signed [31:0] h1_10 = pre1_10[31] ? 32'sd0 : pre1_10;
  wire signed [31:0] pre1_11 = td((-32'sd195)*X) + td(32'sd295*XD) + td(32'sd284*TH) + td((-32'sd108)*THD) + 32'sd12;
  wire signed [31:0] h1_11 = pre1_11[31] ? 32'sd0 : pre1_11;

  // camada 2
  wire signed [31:0] pre2_0 = td((-32'sd73)*h1_0) + td((-32'sd108)*h1_1) + td(32'sd157*h1_2) + td(32'sd199*h1_3) + td((-32'sd76)*h1_4) + td((-32'sd159)*h1_5) + td(32'sd94*h1_6) + td((-32'sd47)*h1_7) + td(32'sd68*h1_8) + td(32'sd53*h1_9) + td((-32'sd25)*h1_10) + td(32'sd173*h1_11) + (-32'sd64);
  wire signed [31:0] h2_0 = pre2_0[31] ? 32'sd0 : pre2_0;
  wire signed [31:0] pre2_1 = td(32'sd29*h1_0) + td((-32'sd53)*h1_1) + td(32'sd111*h1_2) + td((-32'sd163)*h1_3) + td((-32'sd140)*h1_4) + td(32'sd62*h1_5) + td(32'sd14*h1_6) + td((-32'sd116)*h1_7) + td(32'sd187*h1_8) + td((-32'sd25)*h1_9) + td((-32'sd123)*h1_10) + td(32'sd50*h1_11) + (-32'sd26);
  wire signed [31:0] h2_1 = pre2_1[31] ? 32'sd0 : pre2_1;
  wire signed [31:0] pre2_2 = td(32'sd15*h1_0) + td((-32'sd180)*h1_1) + td(32'sd142*h1_2) + td((-32'sd161)*h1_3) + td(32'sd191*h1_4) + td(32'sd85*h1_5) + td(32'sd93*h1_6) + td((-32'sd40)*h1_7) + td((-32'sd26)*h1_8) + td(32'sd155*h1_9) + td((-32'sd21)*h1_10) + td(32'sd104*h1_11) + (-32'sd39);
  wire signed [31:0] h2_2 = pre2_2[31] ? 32'sd0 : pre2_2;
  wire signed [31:0] pre2_3 = td(32'sd54*h1_0) + td(32'sd96*h1_1) + td(32'sd33*h1_2) + td((-32'sd165)*h1_3) + td((-32'sd153)*h1_4) + td((-32'sd62)*h1_5) + td(32'sd42*h1_6) + td(32'sd156*h1_7) + td(32'sd140*h1_8) + td((-32'sd167)*h1_9) + td((-32'sd169)*h1_10) + td(32'sd174*h1_11) + 32'sd29;
  wire signed [31:0] h2_3 = pre2_3[31] ? 32'sd0 : pre2_3;
  wire signed [31:0] pre2_4 = td(32'sd159*h1_0) + td((-32'sd42)*h1_1) + td(32'sd131*h1_2) + td(32'sd95*h1_3) + td(32'sd148*h1_4) + td(32'sd28*h1_5) + td((-32'sd55)*h1_6) + td(32'sd166*h1_7) + td((-32'sd3)*h1_8) + td(32'sd142*h1_9) + td((-32'sd23)*h1_10) + td((-32'sd189)*h1_11) + (-32'sd58);
  wire signed [31:0] h2_4 = pre2_4[31] ? 32'sd0 : pre2_4;
  wire signed [31:0] pre2_5 = td(32'sd36*h1_0) + td((-32'sd19)*h1_1) + td((-32'sd114)*h1_2) + td(32'sd112*h1_3) + td((-32'sd141)*h1_4) + td(32'sd52*h1_5) + td((-32'sd170)*h1_6) + td((-32'sd89)*h1_7) + td(32'sd193*h1_8) + td((-32'sd53)*h1_9) + td((-32'sd134)*h1_10) + td(32'sd178*h1_11) + (-32'sd46);
  wire signed [31:0] h2_5 = pre2_5[31] ? 32'sd0 : pre2_5;
  wire signed [31:0] pre2_6 = td((-32'sd74)*h1_0) + td(32'sd3*h1_1) + td(32'sd0*h1_2) + td(32'sd54*h1_3) + td((-32'sd159)*h1_4) + td((-32'sd115)*h1_5) + td(32'sd29*h1_6) + td(32'sd5*h1_7) + td(32'sd81*h1_8) + td((-32'sd58)*h1_9) + td((-32'sd130)*h1_10) + td(32'sd20*h1_11) + (-32'sd11);
  wire signed [31:0] h2_6 = pre2_6[31] ? 32'sd0 : pre2_6;
  wire signed [31:0] pre2_7 = td(32'sd81*h1_0) + td((-32'sd58)*h1_1) + td(32'sd161*h1_2) + td(32'sd12*h1_3) + td((-32'sd17)*h1_4) + td(32'sd149*h1_5) + td((-32'sd6)*h1_6) + td((-32'sd82)*h1_7) + td((-32'sd123)*h1_8) + td((-32'sd158)*h1_9) + td((-32'sd110)*h1_10) + td((-32'sd123)*h1_11) + 32'sd32;
  wire signed [31:0] h2_7 = pre2_7[31] ? 32'sd0 : pre2_7;
  wire signed [31:0] pre2_8 = td((-32'sd82)*h1_0) + td(32'sd137*h1_1) + td((-32'sd81)*h1_2) + td((-32'sd194)*h1_3) + td(32'sd48*h1_4) + td(32'sd101*h1_5) + td((-32'sd107)*h1_6) + td((-32'sd66)*h1_7) + td((-32'sd56)*h1_8) + td((-32'sd198)*h1_9) + td((-32'sd126)*h1_10) + td(32'sd14*h1_11) + (-32'sd26);
  wire signed [31:0] h2_8 = pre2_8[31] ? 32'sd0 : pre2_8;
  wire signed [31:0] pre2_9 = td(32'sd73*h1_0) + td((-32'sd11)*h1_1) + td(32'sd112*h1_2) + td(32'sd89*h1_3) + td((-32'sd37)*h1_4) + td((-32'sd136)*h1_5) + td(32'sd153*h1_6) + td(32'sd63*h1_7) + td(32'sd116*h1_8) + td(32'sd135*h1_9) + td(32'sd146*h1_10) + td(32'sd178*h1_11) + 32'sd0;
  wire signed [31:0] h2_9 = pre2_9[31] ? 32'sd0 : pre2_9;
  wire signed [31:0] pre2_10 = td((-32'sd173)*h1_0) + td(32'sd33*h1_1) + td(32'sd199*h1_2) + td(32'sd148*h1_3) + td(32'sd86*h1_4) + td(32'sd0*h1_5) + td(32'sd3*h1_6) + td(32'sd4*h1_7) + td(32'sd1*h1_8) + td((-32'sd147)*h1_9) + td(32'sd46*h1_10) + td(32'sd124*h1_11) + 32'sd24;
  wire signed [31:0] h2_10 = pre2_10[31] ? 32'sd0 : pre2_10;
  wire signed [31:0] pre2_11 = td(32'sd5*h1_0) + td((-32'sd169)*h1_1) + td((-32'sd103)*h1_2) + td((-32'sd166)*h1_3) + td((-32'sd94)*h1_4) + td(32'sd25*h1_5) + td((-32'sd117)*h1_6) + td((-32'sd144)*h1_7) + td((-32'sd26)*h1_8) + td(32'sd107*h1_9) + td((-32'sd174)*h1_10) + td((-32'sd148)*h1_11) + 32'sd29;
  wire signed [31:0] h2_11 = pre2_11[31] ? 32'sd0 : pre2_11;

  // saida
  wire signed [31:0] z = td(32'sd42*h2_0) + td((-32'sd138)*h2_1) + td((-32'sd141)*h2_2) + td(32'sd49*h2_3) + td(32'sd38*h2_4) + td(32'sd45*h2_5) + td(32'sd47*h2_6) + td((-32'sd41)*h2_7) + td((-32'sd157)*h2_8) + td((-32'sd127)*h2_9) + td((-32'sd148)*h2_10) + td(32'sd183*h2_11) + 32'sd23;

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

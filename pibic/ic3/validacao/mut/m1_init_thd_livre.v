// Gerado por gen_transition_system.py — NAO EDITAR A MAO
// Actor DDPG real: 4 -> 24 -> 24 -> 1, Q8.8, 745 parametros
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
  wire signed [31:0] pre1_0 = td(32'sd87*X) + td(32'sd134*XD) + td(32'sd1131*TH) + td(32'sd468*THD) + (-32'sd289);
  wire signed [31:0] h1_0 = pre1_0[31] ? 32'sd0 : pre1_0;
  wire signed [31:0] pre1_1 = td((-32'sd139)*X) + td((-32'sd141)*XD) + td((-32'sd474)*TH) + td((-32'sd201)*THD) + 32'sd366;
  wire signed [31:0] h1_1 = pre1_1[31] ? 32'sd0 : pre1_1;
  wire signed [31:0] pre1_2 = td(32'sd1*X) + td((-32'sd122)*XD) + td(32'sd63*TH) + td((-32'sd50)*THD) + (-32'sd315);
  wire signed [31:0] h1_2 = pre1_2[31] ? 32'sd0 : pre1_2;
  wire signed [31:0] pre1_3 = td(32'sd163*X) + td(32'sd186*XD) + td((-32'sd8)*TH) + td(32'sd179*THD) + (-32'sd355);
  wire signed [31:0] h1_3 = pre1_3[31] ? 32'sd0 : pre1_3;
  wire signed [31:0] pre1_4 = td(32'sd68*X) + td((-32'sd52)*XD) + td((-32'sd218)*TH) + td((-32'sd3)*THD) + (-32'sd207);
  wire signed [31:0] h1_4 = pre1_4[31] ? 32'sd0 : pre1_4;
  wire signed [31:0] pre1_5 = td((-32'sd14)*X) + td(32'sd22*XD) + td((-32'sd88)*TH) + td(32'sd32*THD) + (-32'sd57);
  wire signed [31:0] h1_5 = pre1_5[31] ? 32'sd0 : pre1_5;
  wire signed [31:0] pre1_6 = td(32'sd128*X) + td((-32'sd50)*XD) + td((-32'sd34)*TH) + td((-32'sd16)*THD) + 32'sd257;
  wire signed [31:0] h1_6 = pre1_6[31] ? 32'sd0 : pre1_6;
  wire signed [31:0] pre1_7 = td(32'sd44*X) + td((-32'sd79)*XD) + td(32'sd151*TH) + td((-32'sd94)*THD) + (-32'sd139);
  wire signed [31:0] h1_7 = pre1_7[31] ? 32'sd0 : pre1_7;
  wire signed [31:0] pre1_8 = td(32'sd98*X) + td(32'sd0*XD) + td(32'sd236*TH) + td(32'sd28*THD) + (-32'sd291);
  wire signed [31:0] h1_8 = pre1_8[31] ? 32'sd0 : pre1_8;
  wire signed [31:0] pre1_9 = td(32'sd84*X) + td(32'sd14*XD) + td((-32'sd210)*TH) + td(32'sd2*THD) + (-32'sd238);
  wire signed [31:0] h1_9 = pre1_9[31] ? 32'sd0 : pre1_9;
  wire signed [31:0] pre1_10 = td(32'sd67*X) + td(32'sd90*XD) + td(32'sd420*TH) + td(32'sd152*THD) + (-32'sd105);
  wire signed [31:0] h1_10 = pre1_10[31] ? 32'sd0 : pre1_10;
  wire signed [31:0] pre1_11 = td((-32'sd9)*X) + td((-32'sd115)*XD) + td((-32'sd46)*TH) + td((-32'sd115)*THD) + (-32'sd169);
  wire signed [31:0] h1_11 = pre1_11[31] ? 32'sd0 : pre1_11;
  wire signed [31:0] pre1_12 = td(32'sd53*X) + td((-32'sd67)*XD) + td((-32'sd104)*TH) + td((-32'sd66)*THD) + (-32'sd139);
  wire signed [31:0] h1_12 = pre1_12[31] ? 32'sd0 : pre1_12;
  wire signed [31:0] pre1_13 = td(32'sd35*X) + td(32'sd113*XD) + td(32'sd1149*TH) + td(32'sd342*THD) + 32'sd67;
  wire signed [31:0] h1_13 = pre1_13[31] ? 32'sd0 : pre1_13;
  wire signed [31:0] pre1_14 = td(32'sd48*X) + td(32'sd219*XD) + td((-32'sd610)*TH) + td(32'sd101*THD) + (-32'sd44);
  wire signed [31:0] h1_14 = pre1_14[31] ? 32'sd0 : pre1_14;
  wire signed [31:0] pre1_15 = td(32'sd72*X) + td((-32'sd193)*XD) + td(32'sd631*TH) + td((-32'sd28)*THD) + (-32'sd35);
  wire signed [31:0] h1_15 = pre1_15[31] ? 32'sd0 : pre1_15;
  wire signed [31:0] pre1_16 = td((-32'sd135)*X) + td((-32'sd47)*XD) + td(32'sd795*TH) + td(32'sd52*THD) + 32'sd3;
  wire signed [31:0] h1_16 = pre1_16[31] ? 32'sd0 : pre1_16;
  wire signed [31:0] pre1_17 = td(32'sd26*X) + td((-32'sd17)*XD) + td(32'sd564*TH) + td((-32'sd30)*THD) + 32'sd213;
  wire signed [31:0] h1_17 = pre1_17[31] ? 32'sd0 : pre1_17;
  wire signed [31:0] pre1_18 = td(32'sd94*X) + td((-32'sd67)*XD) + td((-32'sd416)*TH) + td((-32'sd55)*THD) + (-32'sd281);
  wire signed [31:0] h1_18 = pre1_18[31] ? 32'sd0 : pre1_18;
  wire signed [31:0] pre1_19 = td((-32'sd407)*X) + td((-32'sd303)*XD) + td((-32'sd893)*TH) + td((-32'sd298)*THD) + (-32'sd317);
  wire signed [31:0] h1_19 = pre1_19[31] ? 32'sd0 : pre1_19;
  wire signed [31:0] pre1_20 = td((-32'sd498)*X) + td((-32'sd453)*XD) + td((-32'sd393)*TH) + td((-32'sd276)*THD) + (-32'sd770);
  wire signed [31:0] h1_20 = pre1_20[31] ? 32'sd0 : pre1_20;
  wire signed [31:0] pre1_21 = td(32'sd157*X) + td(32'sd133*XD) + td((-32'sd111)*TH) + td((-32'sd35)*THD) + (-32'sd296);
  wire signed [31:0] h1_21 = pre1_21[31] ? 32'sd0 : pre1_21;
  wire signed [31:0] pre1_22 = td((-32'sd353)*X) + td((-32'sd315)*XD) + td((-32'sd79)*TH) + td((-32'sd217)*THD) + (-32'sd74);
  wire signed [31:0] h1_22 = pre1_22[31] ? 32'sd0 : pre1_22;
  wire signed [31:0] pre1_23 = td(32'sd92*X) + td(32'sd60*XD) + td((-32'sd580)*TH) + td((-32'sd132)*THD) + (-32'sd31);
  wire signed [31:0] h1_23 = pre1_23[31] ? 32'sd0 : pre1_23;

  // camada 2
  wire signed [31:0] pre2_0 = td((-32'sd142)*h1_0) + td((-32'sd77)*h1_1) + td((-32'sd11)*h1_2) + td((-32'sd35)*h1_3) + td((-32'sd36)*h1_4) + td((-32'sd68)*h1_5) + td(32'sd17*h1_6) + td((-32'sd41)*h1_7) + td((-32'sd73)*h1_8) + td((-32'sd48)*h1_9) + td(32'sd31*h1_10) + td((-32'sd30)*h1_11) + td((-32'sd44)*h1_12) + td(32'sd36*h1_13) + td((-32'sd20)*h1_14) + td((-32'sd174)*h1_15) + td((-32'sd213)*h1_16) + td((-32'sd95)*h1_17) + td(32'sd21*h1_18) + td((-32'sd50)*h1_19) + td((-32'sd55)*h1_20) + td((-32'sd123)*h1_21) + td(32'sd55*h1_22) + td((-32'sd65)*h1_23) + 32'sd34;
  wire signed [31:0] h2_0 = pre2_0[31] ? 32'sd0 : pre2_0;
  wire signed [31:0] pre2_1 = td((-32'sd24)*h1_0) + td((-32'sd14)*h1_1) + td((-32'sd121)*h1_2) + td(32'sd73*h1_3) + td((-32'sd22)*h1_4) + td((-32'sd16)*h1_5) + td(32'sd284*h1_6) + td((-32'sd14)*h1_7) + td(32'sd1*h1_8) + td(32'sd1203*h1_9) + td(32'sd319*h1_10) + td((-32'sd268)*h1_11) + td(32'sd23*h1_12) + td(32'sd249*h1_13) + td((-32'sd145)*h1_14) + td((-32'sd61)*h1_15) + td(32'sd59*h1_16) + td(32'sd483*h1_17) + td(32'sd341*h1_18) + td((-32'sd867)*h1_19) + td((-32'sd1588)*h1_20) + td(32'sd340*h1_21) + td((-32'sd1406)*h1_22) + td(32'sd122*h1_23) + (-32'sd59);
  wire signed [31:0] h2_1 = pre2_1[31] ? 32'sd0 : pre2_1;
  wire signed [31:0] pre2_2 = td(32'sd53*h1_0) + td((-32'sd455)*h1_1) + td((-32'sd132)*h1_2) + td((-32'sd332)*h1_3) + td((-32'sd105)*h1_4) + td((-32'sd5)*h1_5) + td((-32'sd1156)*h1_6) + td((-32'sd16)*h1_7) + td(32'sd44*h1_8) + td((-32'sd904)*h1_9) + td(32'sd768*h1_10) + td((-32'sd375)*h1_11) + td(32'sd27*h1_12) + td((-32'sd135)*h1_13) + td(32'sd216*h1_14) + td(32'sd602*h1_15) + td((-32'sd550)*h1_16) + td((-32'sd7)*h1_17) + td((-32'sd120)*h1_18) + td(32'sd88*h1_19) + td(32'sd254*h1_20) + td((-32'sd343)*h1_21) + td(32'sd379*h1_22) + td((-32'sd1014)*h1_23) + (-32'sd370);
  wire signed [31:0] h2_2 = pre2_2[31] ? 32'sd0 : pre2_2;
  wire signed [31:0] pre2_3 = td((-32'sd116)*h1_0) + td((-32'sd64)*h1_1) + td((-32'sd416)*h1_2) + td((-32'sd593)*h1_3) + td((-32'sd133)*h1_4) + td(32'sd21*h1_5) + td((-32'sd126)*h1_6) + td(32'sd21*h1_7) + td((-32'sd471)*h1_8) + td(32'sd760*h1_9) + td((-32'sd1738)*h1_10) + td(32'sd217*h1_11) + td(32'sd6*h1_12) + td(32'sd536*h1_13) + td((-32'sd186)*h1_14) + td((-32'sd110)*h1_15) + td((-32'sd421)*h1_16) + td(32'sd546*h1_17) + td((-32'sd134)*h1_18) + td(32'sd178*h1_19) + td(32'sd277*h1_20) + td((-32'sd337)*h1_21) + td((-32'sd470)*h1_22) + td(32'sd517*h1_23) + (-32'sd106);
  wire signed [31:0] h2_3 = pre2_3[31] ? 32'sd0 : pre2_3;
  wire signed [31:0] pre2_4 = td((-32'sd20)*h1_0) + td((-32'sd37)*h1_1) + td(32'sd12*h1_2) + td((-32'sd20)*h1_3) + td((-32'sd34)*h1_4) + td(32'sd24*h1_5) + td((-32'sd38)*h1_6) + td((-32'sd10)*h1_7) + td((-32'sd69)*h1_8) + td(32'sd26*h1_9) + td(32'sd7*h1_10) + td(32'sd0*h1_11) + td((-32'sd36)*h1_12) + td((-32'sd48)*h1_13) + td((-32'sd79)*h1_14) + td(32'sd12*h1_15) + td(32'sd20*h1_16) + td(32'sd14*h1_17) + td(32'sd26*h1_18) + td((-32'sd8)*h1_19) + td((-32'sd20)*h1_20) + td((-32'sd47)*h1_21) + td((-32'sd40)*h1_22) + td((-32'sd53)*h1_23) + (-32'sd2);
  wire signed [31:0] h2_4 = pre2_4[31] ? 32'sd0 : pre2_4;
  wire signed [31:0] pre2_5 = td(32'sd283*h1_0) + td(32'sd58*h1_1) + td((-32'sd29)*h1_2) + td(32'sd406*h1_3) + td((-32'sd140)*h1_4) + td(32'sd3*h1_5) + td((-32'sd222)*h1_6) + td(32'sd20*h1_7) + td((-32'sd60)*h1_8) + td((-32'sd2211)*h1_9) + td(32'sd1594*h1_10) + td(32'sd247*h1_11) + td(32'sd5*h1_12) + td((-32'sd670)*h1_13) + td(32'sd387*h1_14) + td(32'sd119*h1_15) + td(32'sd78*h1_16) + td(32'sd208*h1_17) + td(32'sd96*h1_18) + td(32'sd302*h1_19) + td(32'sd194*h1_20) + td(32'sd256*h1_21) + td(32'sd514*h1_22) + td((-32'sd1530)*h1_23) + (-32'sd218);
  wire signed [31:0] h2_5 = pre2_5[31] ? 32'sd0 : pre2_5;
  wire signed [31:0] pre2_6 = td((-32'sd2)*h1_0) + td(32'sd107*h1_1) + td((-32'sd511)*h1_2) + td(32'sd71*h1_3) + td((-32'sd229)*h1_4) + td((-32'sd21)*h1_5) + td((-32'sd489)*h1_6) + td((-32'sd46)*h1_7) + td((-32'sd407)*h1_8) + td(32'sd1164*h1_9) + td((-32'sd1739)*h1_10) + td(32'sd270*h1_11) + td((-32'sd42)*h1_12) + td(32'sd7*h1_13) + td((-32'sd178)*h1_14) + td(32'sd14*h1_15) + td((-32'sd302)*h1_16) + td(32'sd396*h1_17) + td((-32'sd378)*h1_18) + td(32'sd111*h1_19) + td(32'sd234*h1_20) + td(32'sd242*h1_21) + td((-32'sd438)*h1_22) + td(32'sd284*h1_23) + (-32'sd402);
  wire signed [31:0] h2_6 = pre2_6[31] ? 32'sd0 : pre2_6;
  wire signed [31:0] pre2_7 = td(32'sd23*h1_0) + td((-32'sd23)*h1_1) + td((-32'sd6)*h1_2) + td((-32'sd29)*h1_3) + td((-32'sd47)*h1_4) + td((-32'sd51)*h1_5) + td(32'sd1*h1_6) + td((-32'sd29)*h1_7) + td((-32'sd34)*h1_8) + td((-32'sd18)*h1_9) + td((-32'sd18)*h1_10) + td((-32'sd25)*h1_11) + td(32'sd2*h1_12) + td((-32'sd30)*h1_13) + td((-32'sd27)*h1_14) + td((-32'sd19)*h1_15) + td((-32'sd59)*h1_16) + td(32'sd12*h1_17) + td((-32'sd25)*h1_18) + td(32'sd5*h1_19) + td((-32'sd26)*h1_20) + td((-32'sd6)*h1_21) + td(32'sd37*h1_22) + td(32'sd13*h1_23) + (-32'sd76);
  wire signed [31:0] h2_7 = pre2_7[31] ? 32'sd0 : pre2_7;
  wire signed [31:0] pre2_8 = td((-32'sd65)*h1_0) + td((-32'sd2)*h1_1) + td(32'sd27*h1_2) + td(32'sd14*h1_3) + td(32'sd23*h1_4) + td(32'sd46*h1_5) + td((-32'sd8)*h1_6) + td(32'sd11*h1_7) + td((-32'sd52)*h1_8) + td((-32'sd11)*h1_9) + td(32'sd24*h1_10) + td(32'sd18*h1_11) + td((-32'sd33)*h1_12) + td((-32'sd57)*h1_13) + td((-32'sd45)*h1_14) + td((-32'sd57)*h1_15) + td((-32'sd56)*h1_16) + td((-32'sd20)*h1_17) + td(32'sd22*h1_18) + td((-32'sd14)*h1_19) + td((-32'sd45)*h1_20) + td((-32'sd42)*h1_21) + td(32'sd10*h1_22) + td((-32'sd39)*h1_23) + (-32'sd9);
  wire signed [31:0] h2_8 = pre2_8[31] ? 32'sd0 : pre2_8;
  wire signed [31:0] pre2_9 = td(32'sd153*h1_0) + td(32'sd63*h1_1) + td(32'sd263*h1_2) + td(32'sd313*h1_3) + td(32'sd167*h1_4) + td((-32'sd20)*h1_5) + td((-32'sd43)*h1_6) + td((-32'sd47)*h1_7) + td((-32'sd357)*h1_8) + td(32'sd787*h1_9) + td((-32'sd423)*h1_10) + td((-32'sd978)*h1_11) + td((-32'sd19)*h1_12) + td(32'sd149*h1_13) + td(32'sd91*h1_14) + td((-32'sd128)*h1_15) + td(32'sd330*h1_16) + td((-32'sd59)*h1_17) + td(32'sd389*h1_18) + td((-32'sd306)*h1_19) + td((-32'sd173)*h1_20) + td((-32'sd97)*h1_21) + td(32'sd435*h1_22) + td((-32'sd56)*h1_23) + 32'sd79;
  wire signed [31:0] h2_9 = pre2_9[31] ? 32'sd0 : pre2_9;
  wire signed [31:0] pre2_10 = td((-32'sd61)*h1_0) + td((-32'sd174)*h1_1) + td((-32'sd140)*h1_2) + td((-32'sd57)*h1_3) + td((-32'sd120)*h1_4) + td(32'sd28*h1_5) + td((-32'sd326)*h1_6) + td((-32'sd37)*h1_7) + td(32'sd445*h1_8) + td((-32'sd1668)*h1_9) + td(32'sd212*h1_10) + td((-32'sd33)*h1_11) + td((-32'sd16)*h1_12) + td(32'sd492*h1_13) + td(32'sd75*h1_14) + td(32'sd296*h1_15) + td(32'sd191*h1_16) + td(32'sd609*h1_17) + td(32'sd252*h1_18) + td(32'sd235*h1_19) + td(32'sd145*h1_20) + td(32'sd208*h1_21) + td(32'sd481*h1_22) + td(32'sd45*h1_23) + (-32'sd110);
  wire signed [31:0] h2_10 = pre2_10[31] ? 32'sd0 : pre2_10;
  wire signed [31:0] pre2_11 = td((-32'sd125)*h1_0) + td(32'sd74*h1_1) + td(32'sd300*h1_2) + td((-32'sd126)*h1_3) + td(32'sd156*h1_4) + td(32'sd17*h1_5) + td((-32'sd31)*h1_6) + td((-32'sd8)*h1_7) + td((-32'sd460)*h1_8) + td(32'sd1050*h1_9) + td((-32'sd1289)*h1_10) + td((-32'sd194)*h1_11) + td((-32'sd42)*h1_12) + td((-32'sd1173)*h1_13) + td(32'sd331*h1_14) + td(32'sd31*h1_15) + td((-32'sd84)*h1_16) + td((-32'sd435)*h1_17) + td(32'sd91*h1_18) + td(32'sd227*h1_19) + td(32'sd298*h1_20) + td((-32'sd274)*h1_21) + td(32'sd7*h1_22) + td(32'sd276*h1_23) + (-32'sd53);
  wire signed [31:0] h2_11 = pre2_11[31] ? 32'sd0 : pre2_11;
  wire signed [31:0] pre2_12 = td(32'sd139*h1_0) + td(32'sd571*h1_1) + td(32'sd239*h1_2) + td(32'sd139*h1_3) + td(32'sd205*h1_4) + td(32'sd21*h1_5) + td(32'sd434*h1_6) + td((-32'sd25)*h1_7) + td((-32'sd500)*h1_8) + td(32'sd699*h1_9) + td((-32'sd57)*h1_10) + td((-32'sd1089)*h1_11) + td(32'sd7*h1_12) + td(32'sd37*h1_13) + td((-32'sd6)*h1_14) + td((-32'sd209)*h1_15) + td(32'sd59*h1_16) + td(32'sd121*h1_17) + td(32'sd158*h1_18) + td(32'sd204*h1_19) + td((-32'sd412)*h1_20) + td((-32'sd162)*h1_21) + td(32'sd276*h1_22) + td(32'sd129*h1_23) + 32'sd280;
  wire signed [31:0] h2_12 = pre2_12[31] ? 32'sd0 : pre2_12;
  wire signed [31:0] pre2_13 = td((-32'sd595)*h1_0) + td(32'sd399*h1_1) + td((-32'sd160)*h1_2) + td((-32'sd708)*h1_3) + td((-32'sd157)*h1_4) + td(32'sd0*h1_5) + td((-32'sd63)*h1_6) + td(32'sd8*h1_7) + td(32'sd499*h1_8) + td((-32'sd1297)*h1_9) + td((-32'sd471)*h1_10) + td(32'sd356*h1_11) + td(32'sd50*h1_12) + td(32'sd319*h1_13) + td((-32'sd829)*h1_14) + td(32'sd400*h1_15) + td(32'sd208*h1_16) + td(32'sd217*h1_17) + td((-32'sd96)*h1_18) + td(32'sd222*h1_19) + td((-32'sd87)*h1_20) + td((-32'sd28)*h1_21) + td((-32'sd610)*h1_22) + td((-32'sd770)*h1_23) + 32'sd64;
  wire signed [31:0] h2_13 = pre2_13[31] ? 32'sd0 : pre2_13;
  wire signed [31:0] pre2_14 = td(32'sd57*h1_0) + td((-32'sd37)*h1_1) + td((-32'sd308)*h1_2) + td((-32'sd4)*h1_3) + td((-32'sd169)*h1_4) + td(32'sd20*h1_5) + td((-32'sd487)*h1_6) + td((-32'sd35)*h1_7) + td(32'sd382*h1_8) + td((-32'sd912)*h1_9) + td((-32'sd729)*h1_10) + td((-32'sd265)*h1_11) + td(32'sd52*h1_12) + td(32'sd759*h1_13) + td((-32'sd374)*h1_14) + td(32'sd144*h1_15) + td(32'sd6*h1_16) + td(32'sd312*h1_17) + td((-32'sd134)*h1_18) + td((-32'sd71)*h1_19) + td(32'sd190*h1_20) + td(32'sd360*h1_21) + td(32'sd654*h1_22) + td((-32'sd1179)*h1_23) + (-32'sd36);
  wire signed [31:0] h2_14 = pre2_14[31] ? 32'sd0 : pre2_14;
  wire signed [31:0] pre2_15 = td((-32'sd33)*h1_0) + td((-32'sd373)*h1_1) + td((-32'sd336)*h1_2) + td((-32'sd57)*h1_3) + td((-32'sd184)*h1_4) + td((-32'sd27)*h1_5) + td((-32'sd296)*h1_6) + td((-32'sd8)*h1_7) + td(32'sd357*h1_8) + td((-32'sd1279)*h1_9) + td((-32'sd189)*h1_10) + td((-32'sd377)*h1_11) + td(32'sd14*h1_12) + td(32'sd263*h1_13) + td((-32'sd500)*h1_14) + td(32'sd231*h1_15) + td(32'sd247*h1_16) + td(32'sd233*h1_17) + td((-32'sd97)*h1_18) + td((-32'sd255)*h1_19) + td((-32'sd3)*h1_20) + td((-32'sd21)*h1_21) + td(32'sd384*h1_22) + td((-32'sd630)*h1_23) + (-32'sd112);
  wire signed [31:0] h2_15 = pre2_15[31] ? 32'sd0 : pre2_15;
  wire signed [31:0] pre2_16 = td(32'sd185*h1_0) + td((-32'sd640)*h1_1) + td(32'sd50*h1_2) + td(32'sd212*h1_3) + td(32'sd141*h1_4) + td((-32'sd16)*h1_5) + td((-32'sd53)*h1_6) + td((-32'sd43)*h1_7) + td((-32'sd55)*h1_8) + td(32'sd1346*h1_9) + td((-32'sd831)*h1_10) + td((-32'sd849)*h1_11) + td(32'sd23*h1_12) + td((-32'sd530)*h1_13) + td((-32'sd54)*h1_14) + td((-32'sd187)*h1_15) + td(32'sd142*h1_16) + td(32'sd56*h1_17) + td(32'sd172*h1_18) + td(32'sd169*h1_19) + td((-32'sd373)*h1_20) + td((-32'sd69)*h1_21) + td((-32'sd839)*h1_22) + td((-32'sd440)*h1_23) + 32'sd135;
  wire signed [31:0] h2_16 = pre2_16[31] ? 32'sd0 : pre2_16;
  wire signed [31:0] pre2_17 = td((-32'sd120)*h1_0) + td((-32'sd77)*h1_1) + td(32'sd387*h1_2) + td(32'sd35*h1_3) + td(32'sd220*h1_4) + td(32'sd49*h1_5) + td(32'sd568*h1_6) + td(32'sd12*h1_7) + td((-32'sd691)*h1_8) + td(32'sd1524*h1_9) + td((-32'sd714)*h1_10) + td(32'sd97*h1_11) + td((-32'sd31)*h1_12) + td((-32'sd736)*h1_13) + td((-32'sd713)*h1_14) + td((-32'sd490)*h1_15) + td(32'sd3*h1_16) + td((-32'sd1132)*h1_17) + td(32'sd243*h1_18) + td(32'sd791*h1_19) + td(32'sd350*h1_20) + td((-32'sd301)*h1_21) + td((-32'sd112)*h1_22) + td(32'sd919*h1_23) + (-32'sd427);
  wire signed [31:0] h2_17 = pre2_17[31] ? 32'sd0 : pre2_17;
  wire signed [31:0] pre2_18 = td((-32'sd54)*h1_0) + td((-32'sd135)*h1_1) + td((-32'sd72)*h1_2) + td((-32'sd660)*h1_3) + td((-32'sd126)*h1_4) + td(32'sd22*h1_5) + td((-32'sd592)*h1_6) + td(32'sd54*h1_7) + td(32'sd168*h1_8) + td((-32'sd1764)*h1_9) + td(32'sd462*h1_10) + td((-32'sd786)*h1_11) + td((-32'sd23)*h1_12) + td(32'sd172*h1_13) + td((-32'sd635)*h1_14) + td(32'sd172*h1_15) + td((-32'sd110)*h1_16) + td(32'sd123*h1_17) + td(32'sd111*h1_18) + td((-32'sd132)*h1_19) + td((-32'sd98)*h1_20) + td(32'sd106*h1_21) + td(32'sd42*h1_22) + td((-32'sd729)*h1_23) + (-32'sd53);
  wire signed [31:0] h2_18 = pre2_18[31] ? 32'sd0 : pre2_18;
  wire signed [31:0] pre2_19 = td((-32'sd250)*h1_0) + td((-32'sd309)*h1_1) + td((-32'sd496)*h1_2) + td((-32'sd361)*h1_3) + td((-32'sd2)*h1_4) + td((-32'sd30)*h1_5) + td((-32'sd445)*h1_6) + td(32'sd43*h1_7) + td((-32'sd30)*h1_8) + td(32'sd994*h1_9) + td((-32'sd1467)*h1_10) + td((-32'sd116)*h1_11) + td(32'sd13*h1_12) + td((-32'sd164)*h1_13) + td((-32'sd102)*h1_14) + td(32'sd370*h1_15) + td((-32'sd356)*h1_16) + td(32'sd469*h1_17) + td((-32'sd24)*h1_18) + td(32'sd2*h1_19) + td(32'sd252*h1_20) + td(32'sd130*h1_21) + td((-32'sd712)*h1_22) + td(32'sd364*h1_23) + (-32'sd224);
  wire signed [31:0] h2_19 = pre2_19[31] ? 32'sd0 : pre2_19;
  wire signed [31:0] pre2_20 = td(32'sd44*h1_0) + td((-32'sd888)*h1_1) + td((-32'sd51)*h1_2) + td((-32'sd352)*h1_3) + td((-32'sd142)*h1_4) + td((-32'sd5)*h1_5) + td((-32'sd1458)*h1_6) + td((-32'sd6)*h1_7) + td(32'sd604*h1_8) + td((-32'sd815)*h1_9) + td(32'sd1192*h1_10) + td((-32'sd46)*h1_11) + td((-32'sd19)*h1_12) + td(32'sd256*h1_13) + td((-32'sd92)*h1_14) + td(32'sd579*h1_15) + td((-32'sd185)*h1_16) + td(32'sd103*h1_17) + td(32'sd78*h1_18) + td(32'sd447*h1_19) + td(32'sd274*h1_20) + td(32'sd515*h1_21) + td(32'sd759*h1_22) + td((-32'sd712)*h1_23) + (-32'sd212);
  wire signed [31:0] h2_20 = pre2_20[31] ? 32'sd0 : pre2_20;
  wire signed [31:0] pre2_21 = td((-32'sd27)*h1_0) + td(32'sd54*h1_1) + td(32'sd70*h1_2) + td((-32'sd269)*h1_3) + td((-32'sd83)*h1_4) + td(32'sd13*h1_5) + td(32'sd27*h1_6) + td((-32'sd26)*h1_7) + td(32'sd332*h1_8) + td(32'sd491*h1_9) + td(32'sd665*h1_10) + td(32'sd54*h1_11) + td((-32'sd62)*h1_12) + td(32'sd220*h1_13) + td((-32'sd725)*h1_14) + td(32'sd19*h1_15) + td(32'sd141*h1_16) + td((-32'sd65)*h1_17) + td(32'sd368*h1_18) + td((-32'sd252)*h1_19) + td((-32'sd176)*h1_20) + td(32'sd45*h1_21) + td(32'sd337*h1_22) + td(32'sd157*h1_23) + (-32'sd160);
  wire signed [31:0] h2_21 = pre2_21[31] ? 32'sd0 : pre2_21;
  wire signed [31:0] pre2_22 = td(32'sd122*h1_0) + td(32'sd178*h1_1) + td((-32'sd30)*h1_2) + td((-32'sd49)*h1_3) + td((-32'sd91)*h1_4) + td(32'sd17*h1_5) + td(32'sd59*h1_6) + td(32'sd44*h1_7) + td(32'sd418*h1_8) + td((-32'sd1474)*h1_9) + td(32'sd695*h1_10) + td((-32'sd268)*h1_11) + td((-32'sd34)*h1_12) + td(32'sd297*h1_13) + td((-32'sd198)*h1_14) + td(32'sd35*h1_15) + td(32'sd307*h1_16) + td(32'sd568*h1_17) + td((-32'sd36)*h1_18) + td((-32'sd284)*h1_19) + td((-32'sd428)*h1_20) + td(32'sd148*h1_21) + td(32'sd317*h1_22) + td((-32'sd547)*h1_23) + 32'sd121;
  wire signed [31:0] h2_22 = pre2_22[31] ? 32'sd0 : pre2_22;
  wire signed [31:0] pre2_23 = td((-32'sd627)*h1_0) + td(32'sd193*h1_1) + td((-32'sd133)*h1_2) + td((-32'sd270)*h1_3) + td((-32'sd142)*h1_4) + td(32'sd20*h1_5) + td(32'sd220*h1_6) + td((-32'sd13)*h1_7) + td((-32'sd161)*h1_8) + td((-32'sd1222)*h1_9) + td(32'sd405*h1_10) + td(32'sd155*h1_11) + td(32'sd5*h1_12) + td((-32'sd505)*h1_13) + td((-32'sd125)*h1_14) + td((-32'sd632)*h1_15) + td((-32'sd631)*h1_16) + td(32'sd251*h1_17) + td(32'sd14*h1_18) + td(32'sd139*h1_19) + td(32'sd399*h1_20) + td((-32'sd1117)*h1_21) + td(32'sd612*h1_22) + td((-32'sd834)*h1_23) + (-32'sd83);
  wire signed [31:0] h2_23 = pre2_23[31] ? 32'sd0 : pre2_23;

  // saida
  wire signed [31:0] z = td(32'sd68*h2_0) + td(32'sd342*h2_1) + td(32'sd555*h2_2) + td((-32'sd280)*h2_3) + td(32'sd6*h2_4) + td((-32'sd196)*h2_5) + td((-32'sd201)*h2_6) + td((-32'sd28)*h2_7) + td((-32'sd32)*h2_8) + td((-32'sd496)*h2_9) + td(32'sd240*h2_10) + td((-32'sd200)*h2_11) + td((-32'sd291)*h2_12) + td(32'sd341*h2_13) + td(32'sd747*h2_14) + td(32'sd793*h2_15) + td(32'sd342*h2_16) + td((-32'sd945)*h2_17) + td((-32'sd762)*h2_18) + td((-32'sd209)*h2_19) + td(32'sd423*h2_20) + td(32'sd707*h2_21) + td(32'sd279*h2_22) + td((-32'sd556)*h2_23) + (-32'sd201);

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
      x <= ci(i_x); xd <= ci(i_xd); th <= ci(i_th); thd <= i_thd;
    end else begin
      x <= n_x[15:0];   xd  <= n_xd[15:0];
      th <= n_th[15:0]; thd <= n_thd[15:0];
    end
  end

  // propriedade de seguranca: o pendulo nunca passa de 12 graus
  assign bad = (th > 16'sd53) || (th < -16'sd53);
endmodule

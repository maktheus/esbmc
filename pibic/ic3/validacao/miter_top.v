module miter(input signed [15:0] x, xd, th, thd, output ok);
  wire g_bad, r_bad; wire signed [31:0] gx, gxd, gth, gthd, rx, rxd, rth, rthd;
  top g(.clk(1'b0), .x(x), .xd(xd), .th(th), .thd(thd), .bad(g_bad), .o_nx(gx), .o_nxd(gxd), .o_nth(gth), .o_nthd(gthd));
  refm r(.x(x), .xd(xd), .th(th), .thd(thd), .nx(rx), .nxd(rxd), .nth(rth), .nthd(rthd), .bad(r_bad));
  assign ok = (gx==rx)&&(gxd==rxd)&&(gth==rth)&&(gthd==rthd)&&(g_bad==r_bad);
endmodule

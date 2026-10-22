import fs from 'fs';
const B = '\\';
const s = fs.readFileSync('main_antigo.tex','utf8');
const rd = f => fs.readFileSync('parts/'+f,'utf8');
const i = k => { const n = s.indexOf(k); if(n<0) throw new Error(k); return n; };
const pre = s.slice(0, i(B+'chapter{Introdução}'));
const fund = s.slice(i(B+'chapter{Fundamentação}'), i(B+'chapter{Parte I'));
const mid = s.slice(i(B+'chapter{Parte I'), i(B+'chapter{Conclusão}'));
const post = s.slice(i(B+'begin{apendicesenv}'));
const demote = t => t.split(B+'subsection{').join(B+'subsubsection{').split(B+'section{').join(B+'subsection{').split(B+'chapter{').join(B+'section{');
let m = mid.replace(B+'chapter{Discussão: o que está provado e quão novo é}'+B+'label{cap:disc}', B+'chapter{Resultados preliminares: discussão e ameaças à validade}'+B+'label{sec:discpre}')
  .replace('Parte I: o '+B+'emph{cart-pole} verificado com ESBMC','Rota BMC: o '+B+'emph{cart-pole} verificado com ESBMC')
  .replace('Parte II: a rota IC3/PDR','Rota indutiva: IC3/PDR');
m = demote(m);
const f = fund.replace(B+'chapter{Fundamentação}', B+'chapter{Referencial Teórico}');
const tail = rd('tail.tex').replace('da Seção~'+B+'ref{cap:metod}','do Capítulo~'+B+'ref{cap:metod}');
let out = pre + rd('intro.tex') + '\n' + rd('correlatos.tex') + '\n' + f + '\n' + rd('metodo.tex') + '\n' + m + '\n' + tail + '\n' + post;
out = out.replace(/Capítulo~\\ref\{cap:(esbmc|ic3|valid)\}/g, 'Seção~\\ref{cap:$1}');
const extra = fs.readFileSync('parts/bib.tex','utf8');
out = out.replace(B+'end{thebibliography}', extra + B+'end{thebibliography}');
fs.writeFileSync('main.tex', out);

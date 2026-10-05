# Fundamentação teórica

Este documento deriva, passo a passo, cada equação implementada no pacote `boundary_layer`
e indica o módulo em que ela aparece. A ordem segue a lógica física: das equações de
Navier–Stokes às equações de Prandtl, delas às soluções exatas (Blasius, Falkner–Skan), aos
métodos aproximados (von Kármán–Pohlhausen, Thwaites) e, por fim, ao método numérico que
resolve o caso geral.

## Sumário

1. [Equações de Prandtl](#1-equações-de-prandtl)
2. [Adimensionalização usada no código](#2-adimensionalização-usada-no-código)
3. [Espessuras integrais e atrito](#3-espessuras-integrais-e-atrito)
4. [Solução de similaridade de Blasius](#4-solução-de-similaridade-de-blasius)
5. [Família de Falkner–Skan](#5-família-de-falknerskan)
6. [Equação integral de von Kármán](#6-equação-integral-de-von-kármán)
7. [Perfis de Pohlhausen](#7-perfis-de-pohlhausen)
8. [Método de Thwaites](#8-método-de-thwaites)
9. [Separação](#9-separação)
10. [Método numérico: marcha de Crank–Nicolson](#10-método-numérico-marcha-de-cranknicolson)
11. [Correlações turbulentas (contexto)](#11-correlações-turbulentas-contexto)
12. [Referências](#12-referências)

---

## 1. Equações de Prandtl

Para escoamento 2D, permanente, incompressível e com propriedades constantes, as equações
de Navier–Stokes são

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0,
```

```math
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}
  = -\frac{1}{\rho}\frac{\partial p}{\partial x}
  + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right),
```

```math
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}
  = -\frac{1}{\rho}\frac{\partial p}{\partial y}
  + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right).
```

**Análise de ordem de grandeza.** Seja $L$ o comprimento característico na direção do
escoamento e $\delta \ll L$ a espessura da camada viscosa. Com $u \sim U$, a continuidade
exige $v \sim U\delta/L$. Na equação de $x$, os termos convectivos são $O(U^2/L)$ e o termo
viscoso dominante é $\nu U/\delta^2$; para que inércia e viscosidade se equilibrem,

```math
\frac{\delta}{L} \sim \frac{1}{\sqrt{Re_L}}, \qquad Re_L = \frac{UL}{\nu}.
```

Ao mesmo tempo, $\partial^2 u/\partial x^2 \sim U/L^2$ é $O(Re_L^{-1})$ em relação a
$\partial^2 u/\partial y^2$ e é desprezado. A equação de $y$ reduz-se a
$\partial p/\partial y = O(Re_L^{-1})$: **a pressão é imposta pelo escoamento externo**
através da camada. No bordo, a equação de Euler dá $-\frac{1}{\rho}\frac{dp}{dx} = U_e\frac{dU_e}{dx}$.
Assim chegam-se às equações de Prandtl (1904):

```math
\boxed{\;
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0,\qquad
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}
  = U_e\frac{dU_e}{dx} + \nu\frac{\partial^2 u}{\partial y^2}
\;}
```

com $u = v = 0$ em $y = 0$ (não deslizamento e parede impermeável) e $u \to U_e(x)$ quando
$y \to \infty$.

Duas consequências estruturais orientam todo o projeto:

* **O sistema é parabólico em $x$.** Não há difusão em $x$; a informação viaja somente a
  jusante. Por isso o problema pode ser resolvido por **marcha** a partir de um perfil
  inicial (seção 10), ao contrário de Navier–Stokes, que é elíptico.
* **A pressão deixa de ser incógnita.** $U_e(x)$ é um dado; a única incógnita é $u$, e $v$
  sai diretamente da continuidade.

## 2. Adimensionalização usada no código

Todo o código trabalha com variáveis reescalonadas segundo a própria análise acima:

```math
\tilde x = \frac{x}{L},\quad \tilde y = \frac{y\sqrt{Re_L}}{L},\quad
\tilde u = \frac{u}{U_0},\quad \tilde v = \frac{v\sqrt{Re_L}}{U_0},\quad
\tilde U_e = \frac{U_e}{U_0}.
```

Substituindo, $Re_L$ desaparece por completo:

```math
\tilde u\,\tilde u_{\tilde x} + \tilde v\,\tilde u_{\tilde y}
  = \tilde U_e\tilde U_e' + \tilde u_{\tilde y\tilde y}.
```

Uma única solução escalada vale **para qualquer número de Reynolds**; as grandezas físicas
são recuperadas ao final (é o que a figura 07 faz ao usar a mesma solução para
$Re_L = 10^4, 10^5, 10^6$). Em particular,
$c_f\sqrt{Re_L} = 2\,\tilde\tau_w/\tilde U_e^2$ com $\tilde\tau_w = \partial\tilde u/\partial\tilde y|_0$.

## 3. Espessuras integrais e atrito

```math
\delta^* = \int_0^\infty\left(1 - \frac{u}{U_e}\right)dy,\qquad
\theta = \int_0^\infty \frac{u}{U_e}\left(1 - \frac{u}{U_e}\right)dy,\qquad
H = \frac{\delta^*}{\theta},
```

```math
\tau_w = \mu\left.\frac{\partial u}{\partial y}\right|_{y=0},\qquad
c_f = \frac{2\,\tau_w}{\rho U_e^2}.
```

* $\delta^*$ (deslocamento) é quanto a parede precisaria ser deslocada para que um
  escoamento invíscido transportasse a mesma vazão — e é por isso que as linhas de corrente
  externas na figura 01 sobem levemente.
* $\theta$ (quantidade de movimento) mede o déficit de fluxo de quantidade de movimento.
* $H$ indica a "saúde" do perfil: 2.59 na placa plana, ~4 perto da separação laminar.
* $\delta_{99}$ é a altura em que $u = 0.99\,U_e$: intuitiva, mas arbitrária.

No código: `MarchingResult.delta_star`, `.theta`, `.shape_factor`, `.delta()`, `.cf`.

## 4. Solução de similaridade de Blasius

Na placa plana ($U_e = U_\infty$) não há escala de comprimento natural em $x$; isso sugere
perfis autossemelhantes $u/U_\infty = F(y/\delta(x))$ com $\delta \propto \sqrt{\nu x/U_\infty}$.
Com a função de corrente $\psi$ ($u = \psi_y$, $v = -\psi_x$, satisfazendo a continuidade
identicamente):

```math
\eta = y\sqrt{\frac{U_\infty}{2\nu x}},\qquad \psi = \sqrt{2\nu U_\infty x}\;f(\eta).
```

Substituindo na equação de quantidade de movimento, a EDP vira uma EDO:

```math
\boxed{\,f''' + f\,f'' = 0,\qquad f(0) = f'(0) = 0,\quad f'(\infty) = 1\,}
```

com $u/U_\infty = f'(\eta)$ e
$v\sqrt{Re_x}/U_\infty = (\eta f' - f)/\sqrt{2}$. É um problema de contorno em dois pontos,
resolvido em `similarity.falkner_skan` por colocação (`scipy.integrate.solve_bvp`,
tolerância $10^{-10}$). Resultados (reproduzidos pelos testes):

| Grandeza | Valor | Forma clássica |
|---|---|---|
| $f''(0)$ | 0.469600 | 0.332057 na variável $y\sqrt{U/\nu x}$ |
| $\delta_{99}/x$ | $4.910/\sqrt{Re_x}$ | |
| $\delta^*/x$ | $1.7208/\sqrt{Re_x}$ | |
| $\theta/x$ | $0.6641/\sqrt{Re_x}$ | |
| $c_f$ | $0.6641/\sqrt{Re_x}$ | |
| $H$ | 2.591 | |
| $v_\infty\sqrt{Re_x}/U_\infty$ | 0.8604 | |

Note que $v$ não se anula fora da camada: o crescimento de $\delta^*$ empurra o escoamento
externo para cima.

## 5. Família de Falkner–Skan

A similaridade sobrevive a um gradiente de pressão quando $U_e = C x^m$ (escoamento em
cunha de semiângulo $\beta\pi/2$). Com

```math
\eta = y\sqrt{\frac{(m+1)\,U_e}{2\nu x}},\qquad
\psi = \sqrt{\frac{2\nu U_e x}{m+1}}\;f(\eta),\qquad \beta = \frac{2m}{m+1},
```

obtém-se

```math
\boxed{\,f''' + f\,f'' + \beta\left(1 - f'^2\right) = 0\,}
```

e, derivando $\psi$,

```math
u = U_e f', \qquad
v = -\sqrt{\frac{(m+1)\,\nu U_e}{2x}}\left[f + \frac{m-1}{m+1}\,\eta f'\right].
```

* $\beta = 0$: Blasius. $\beta = 1$: ponto de estagnação de Hiemenz ($f''(0) = 1.2326$).
* $\beta < 0$ (gradiente adverso): o perfil ganha um ponto de inflexão e $f''(0)$ diminui até
  se anular em $\beta = -0.19884$ — **separação incipiente** (Hartree, 1937).
* Para $-0.1988 < \beta < 0$ existem duas soluções; a de interesse físico (colada) é obtida
  por continuação a partir de $\beta = 0$ (`similarity.falkner_skan`).

Na figura 03 essas soluções exatas são comparadas com o solver de marcha aplicado a
escoamentos reais $U_e = x^m$: os dois concordam em $f''(0)$ e em $H$ em toda a faixa.

## 6. Equação integral de von Kármán

Integrando a equação de Prandtl de $y = 0$ a $\infty$ (usando a continuidade para eliminar
$v$) obtém-se uma EDO exata para as espessuras integrais:

```math
\boxed{\,\frac{d\theta}{dx} + (2 + H)\,\frac{\theta}{U_e}\frac{dU_e}{dx} = \frac{\tau_w}{\rho U_e^2} = \frac{c_f}{2}\,}
```

Ela é exata, mas tem três incógnitas ($\theta$, $H$, $\tau_w$). Os métodos integrais
fecham o problema admitindo algo sobre a forma do perfil.

## 7. Perfis de Pohlhausen

Admitindo $u/U = F(s)$, $s = y/\delta$, na placa plana, defina
$\alpha = \int_0^1 F(1-F)\,ds$ ($=\theta/\delta$) e $F'(0)$. A equação integral vira
$\alpha\,\delta\,d\delta/dx = \nu F'(0)/U$, logo

```math
\frac{\delta}{x} = \sqrt{\frac{2F'(0)}{\alpha}}\,\frac{1}{\sqrt{Re_x}},\qquad
c_f\sqrt{Re_x} = \sqrt{2\,\alpha\,F'(0)}.
```

O módulo `integral.py` calcula $\alpha$ por quadratura para qualquer $F$; a figura 04
compara cinco perfis com Blasius. Observações:

* **Grandezas integrais são muito mais robustas que $\delta$.** Até o perfil linear acerta
  $\delta^*$ com 0.7 % de erro, enquanto erra $\delta$ em 29 %. A equação integral impõe
  conservação de quantidade de movimento, não a forma local do perfil.
* O perfil senoidal é o melhor entre os testados em $c_f$ (−1.4 %); o cúbico, o clássico de
  livro-texto, erra −2.7 %.

## 8. Método de Thwaites

Multiplicando a equação integral por $U_e\theta/\nu$ e definindo
$\lambda = \frac{\theta^2}{\nu}\frac{dU_e}{dx}$:

```math
\frac{U_e}{\nu}\frac{d\theta^2}{dx} = 2\left[\,l(\lambda) - (2 + H(\lambda))\,\lambda\,\right] \equiv F(\lambda),
\qquad l = \frac{\tau_w\theta}{\mu U_e}.
```

Thwaites (1949) observou que, para um grande conjunto de soluções exatas,
$F(\lambda) \approx a - b\lambda$ com $a = 0.45$, $b = 6$. A EDO torna-se linear em $\theta^2$ e
integra-se em forma fechada para **qualquer** $U_e(x)$:

```math
\boxed{\,\theta^2(x) = \frac{0.45\,\nu}{U_e^6(x)}\int_0^x U_e^5\,dx'\,}
```

$\tau_w$ e $H$ vêm das correlações $l(\lambda)$, $H(\lambda)$ (Cebeci & Bradshaw, 1977),
implementadas em `integral.thwaites_l` e `integral.thwaites_H`. A separação é prevista
quando $l(\lambda) = 0$, isto é, $\lambda \approx -0.09$.

Para o escoamento de Howarth ($U_e = U_0(1 - x/L)$) a integral é analítica:
$\theta^2 = 0.075\,[(1-x)^{-6} - 1]$ e a separação ocorre em
$x_s = 1 - (1 + 0.0898/0.075)^{-1/6} = 0.1230$ — a 2.6 % do valor exato 0.1199.

## 9. Separação

Em gradiente adverso ($dU_e/dx < 0$) o fluido próximo à parede, já lento, é desacelerado
também pela pressão. Na parede, a equação de Prandtl fornece a **condição de compatibilidade**

```math
\nu\left.\frac{\partial^2 u}{\partial y^2}\right|_{0} = -U_e\frac{dU_e}{dx} = \frac{1}{\rho}\frac{dp}{dx},
```

então, sob gradiente adverso, a curvatura do perfil na parede é positiva enquanto longe dela
é negativa: existe um **ponto de inflexão** (figura 05, painel central). Continuando a jusante,
$\tau_w \to 0$ e o escoamento separa.

As equações de Prandtl tornam-se **singulares** nesse ponto (Goldstein, 1948), com
$\tau_w \propto \sqrt{x_s - x}$. O solver usa esse comportamento: quando $\tau_w$ cai abaixo de um
limiar ou a iteração deixa de convergir, $x_s$ é extrapolado ajustando uma reta a $\tau_w^2(x)$.
Resultado para Howarth: $x_s = 0.1198$, contra 0.1199 (Howarth, 1938) e 0.11988 (Smith, 1977).

## 10. Método numérico: marcha de Crank–Nicolson

Implementado em `marching.solve`. Ideias centrais:

**Malha.** Em $y$, uma malha esticada $y_j = y_{\max}\sinh(a s_j)/\sinh(a)$ concentra pontos na
parede, onde estão os gradientes. Em $x$, estações concentradas perto do bordo de ataque,
onde $\delta \propto \sqrt{x}$ varia mais rápido.

**Discretização em $y$.** Diferenças centradas de três pontos para malha não uniforme
($h_- = y_j - y_{j-1}$, $h_+ = y_{j+1} - y_j$):

```math
\left.\frac{\partial^2 u}{\partial y^2}\right|_j \approx
\frac{2u_{j-1}}{h_-(h_- + h_+)} - \frac{2u_j}{h_- h_+} + \frac{2u_{j+1}}{h_+(h_- + h_+)},
```

```math
\left.\frac{\partial u}{\partial y}\right|_j \approx
-\frac{h_+\,u_{j-1}}{h_-(h_- + h_+)} + \frac{(h_+ - h_-)\,u_j}{h_- h_+} + \frac{h_-\,u_{j+1}}{h_+(h_- + h_+)}.
```

Em uma malha com estiramento suave ambas são de segunda ordem (confirmado: ordem 2.08 na
figura 06). $\tau_w$ é obtido por uma fórmula unilateral de segunda ordem.

**Discretização em $x$ (Crank–Nicolson).** A equação é escrita no ponto médio
$x_{n+1/2}$, com $\bar u = (u^{n+1} + u^n)/2$:

```math
\bar u_j\,\frac{u_j^{n+1} - u_j^n}{\Delta x}
+ \bar v_j\,\delta_y\bar u_j
= \left(U_e U_e'\right)_{n+1/2} + \delta_{yy}\bar u_j .
```

**Continuidade.** A velocidade normal no meio-passo sai da integração exata (trapézios)
da continuidade:

```math
\bar v(y) = -\int_0^y \frac{u^{n+1} - u^n}{\Delta x}\,dy'.
```

**Linearização.** Os coeficientes $\bar u$, $\bar v$ dependem de $u^{n+1}$. Usa-se iteração de
Picard: congela-se os coeficientes, resolve-se o sistema **tridiagonal** resultante
(`scipy.linalg.solve_banded`, $O(N)$), atualizam-se os coeficientes e repete-se até
$\max|\Delta u| < 10^{-10}$. A convergência é linear (~15–20 iterações por estação,
partindo de uma extrapolação linear das duas estações anteriores), mas cada iteração custa
apenas $O(N)$: a placa plana completa (500 × 321 pontos) roda em cerca de 1–2 s. Um Newton
completo convergiria em 3–4 iterações, porém o termo não local $\bar v\,\partial_y\bar u$
tornaria o jacobiano denso — uma troca que não compensa neste tamanho de problema.

**Partida de Rannacher.** O Crank–Nicolson não amortece modos de alta frequência. Se o
perfil inicial não é solução exata do problema a jusante (caso geral, em que se parte da
solução de similaridade local), esse erro inicial contamina a convergência. A figura 06
mostra o efeito: com CN puro a ordem observada em $x$ oscila entre ~1 e ~2.8. A correção
padrão (Rannacher, 1984) é avançar os dois primeiros intervalos com **dois meio-passos de
Euler implícito** cada (fortemente amortecedor); depois disso o esquema recupera a ordem 2
limpa (1.98 medida). Vale notar que, nas malhas testadas, o CN puro tem erro absoluto
menor; o ganho da partida de Rannacher não é precisão bruta, e sim **previsibilidade**: só
com ordem assintótica estável o refino de malha fornece uma estimativa de erro confiável.

**Condição inicial.** Por padrão, o perfil de Falkner–Skan com o expoente local
$m = x\,U_e'/U_e$ em $x_0$ — exato para escoamentos em cunha, ótima aproximação nos demais.

**Validação.** Ver `tests/test_validation.py`: placa plana contra Blasius (erro em $c_f$
< 0.1 %, campo de $u$ com erro < $2\times10^{-4}$), escoamentos em cunha contra Falkner–Skan,
separação de Howarth, e ordem de convergência.

## 11. Correlações turbulentas (contexto)

As equações de Prandtl laminares não descrevem a camada turbulenta: as tensões de
Reynolds $-\rho\overline{u'v'}$ exigem um modelo de fechamento. Para situar a solução laminar no
mapa completo, `turbulent.py` traz correlações empíricas clássicas para placa plana lisa:

```math
\frac{u}{U} = \left(\frac{y}{\delta}\right)^{1/7},\qquad
\frac{\delta}{x} = \frac{0.37}{Re_x^{1/5}},\qquad
c_f = \frac{0.0592}{Re_x^{1/5}},\qquad
c_f = \frac{0.455}{\ln^2(0.06\,Re_x)}.
```

A transição natural costuma ocorrer para $Re_x \sim 3\times10^5$ a $3\times10^6$, dependendo da
turbulência do escoamento livre e da rugosidade. Acima dela o atrito é várias vezes maior
(figura 07).

## 12. Referências

1. Prandtl, L. (1904). Über Flüssigkeitsbewegung bei sehr kleiner Reibung. *Verh. III Int. Math.-Kongr.*, Heidelberg.
2. Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit kleiner Reibung. *Z. Math. Phys.* 56, 1–37.
3. Falkner, V. M. & Skan, S. W. (1931). Some approximate solutions of the boundary layer equations. *Phil. Mag.* 12, 865–896.
4. Hartree, D. R. (1937). On an equation occurring in Falkner and Skan's approximate treatment of the equations of the boundary layer. *Proc. Camb. Phil. Soc.* 33, 223–239.
5. Howarth, L. (1938). On the solution of the laminar boundary layer equations. *Proc. R. Soc. Lond. A* 164, 547–579.
6. Goldstein, S. (1948). On laminar boundary-layer flow near a position of separation. *Q. J. Mech. Appl. Math.* 1, 43–69.
7. Thwaites, B. (1949). Approximate calculation of the laminar boundary layer. *Aeronaut. Q.* 1, 245–280.
8. Cebeci, T. & Bradshaw, P. (1977). *Momentum Transfer in Boundary Layers*. Hemisphere.
9. Smith, F. T. (1977). The laminar separation of an incompressible fluid streaming past a smooth surface. *Proc. R. Soc. Lond. A* 356, 443–463.
10. Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numer. Math.* 43, 309–327.
11. White, F. M. (2006). *Viscous Fluid Flow*, 3ª ed. McGraw-Hill.
12. Schlichting, H. & Gersten, K. (2017). *Boundary-Layer Theory*, 9ª ed. Springer.

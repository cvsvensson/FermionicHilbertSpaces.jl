// Tri-junction used in examples/braiding.jl. Render with `typst compile braiding_junction.typ braiding_junction.svg`.
#set page(width: 384pt, height: 140pt, margin: 0pt, fill: white)
#set text(size: 10pt)

#let w = 128pt
#let c = (64pt, 76pt)
#let r = 44pt
#let at(angle, d) = (c.at(0) + d * calc.cos(angle), c.at(1) - d * calc.sin(angle))
#let pos = (at(90deg, r), at(210deg, r), at(330deg, r))
#let blue = rgb("#1f6fb4")
#let orange = rgb("#d9731a")

#let dot(p, fill) = place(dx: p.at(0) - 5pt, dy: p.at(1) - 5pt, circle(radius: 5pt, fill: fill, stroke: 0.8pt + black))
#let label(p, body) = place(dx: p.at(0) - 15pt, dy: p.at(1) - 8pt, box(width: 30pt, height: 16pt, align(center + horizon, body)))
#let lerp(p, q, s) = (p.at(0) + s * (q.at(0) - p.at(0)), p.at(1) + s * (q.at(1) - p.at(1)))

// The zero mode passes through the junction, close to the centre.
#let hop(from, to, color) = {
  let a = pos.at(from)
  let b = pos.at(to)
  let ctrl = lerp(c, lerp(a, b, 0.5), 0.3)
  let tip = lerp(b, ctrl, 0.2)
  place(curve(stroke: 1.6pt + color, curve.move(lerp(a, ctrl, 0.2)), curve.quad(ctrl, tip)))
  let dir = (tip.at(0) - ctrl.at(0), tip.at(1) - ctrl.at(1))
  let len = calc.sqrt(calc.pow(dir.at(0) / 1pt, 2) + calc.pow(dir.at(1) / 1pt, 2))
  let u = (dir.at(0) / len, dir.at(1) / len)
  let back = (tip.at(0) - 7 * u.at(0), tip.at(1) - 7 * u.at(1))
  let side = (-u.at(1) * 3.5, u.at(0) * 3.5)
  place(polygon(fill: color, tip,
    (back.at(0) + side.at(0), back.at(1) + side.at(1)),
    (back.at(0) - side.at(0), back.at(1) - side.at(1))))
}

// Step n: Δ_n is on at the start and Δ_{n+1} is switched on before Δ_n is switched off,
// which moves a zero mode from γ_{n+1} to γ_n. `modes` gives the colour of each outer Majorana.
#let step(n, modes, mover) = box(width: w, height: 140pt, {
  let on = n - 1
  let next = calc.rem(n, 3)
  for i in range(3) {
    let s = if i == on { 3pt + black } else if i == next { (paint: black, thickness: 2pt, dash: "dashed") } else { 3pt + luma(215) }
    place(line(start: c, end: pos.at(i), stroke: s))
  }
  hop(next, on, mover)
  dot(c, luma(120))
  for i in range(3) { dot(pos.at(i), modes.at(i)) }
  label(lerp(c, pos.at(next), -0.32), $γ_0$)
  label((pos.at(0).at(0) + 16pt, pos.at(0).at(1)), $γ_1$)
  label((pos.at(1).at(0) - 2pt, pos.at(1).at(1) + 13pt), $γ_2$)
  label((pos.at(2).at(0) + 2pt, pos.at(2).at(1) + 13pt), $γ_3$)
  place(dx: 0pt, dy: 6pt, box(width: w, align(center, strong[Step #n])))
  let names = ($Δ_1$, $Δ_2$, $Δ_3$)
  place(dx: 0pt, dy: 124pt, box(width: w, align(center, [#names.at(on) $->$ #names.at(next)])))
})

#step(1, (white, blue, orange), blue)#step(2, (blue, white, orange), orange)#step(3, (blue, orange, white), blue)

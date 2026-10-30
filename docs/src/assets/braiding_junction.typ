// Tri-junction used in examples/braiding.jl. Render with `typst compile braiding_junction.typ braiding_junction.svg`.
#set page(width: 190pt, height: 150pt, margin: 0pt, fill: white)
#set text(size: 11pt)

#let c = (95pt, 82pt)
#let r = 65pt
#let at(angle, d) = (c.at(0) + d * calc.cos(angle), c.at(1) - d * calc.sin(angle))
#let pos = (at(90deg, r), at(210deg, r), at(330deg, r))
#let blue = rgb("#1f6fb4")
#let orange = rgb("#d9731a")

#let dot(p, fill) = place(dx: p.at(0) - 6pt, dy: p.at(1) - 6pt, circle(radius: 6pt, fill: fill, stroke: 0.8pt + black))
#let label(p, body) = place(dx: p.at(0) - 15pt, dy: p.at(1) - 8pt, box(width: 30pt, height: 16pt, align(center + horizon, body)))

// Exchange path: each zero mode passes through the junction, close to the centre.
#let hop(from, to, color, n) = {
  let a = pos.at(from)
  let b = pos.at(to)
  let mid = ((a.at(0) + b.at(0)) / 2, (a.at(1) + b.at(1)) / 2)
  let ctrl = (c.at(0) + 0.35 * (mid.at(0) - c.at(0)), c.at(1) + 0.35 * (mid.at(1) - c.at(1)))
  let shrink(p, q, s) = (p.at(0) + s * (q.at(0) - p.at(0)), p.at(1) + s * (q.at(1) - p.at(1)))
  place(curve(
    stroke: 1.6pt + color,
    curve.move(shrink(a, ctrl, 0.18)),
    curve.quad(ctrl, shrink(b, ctrl, 0.18)),
  ))
  let tip = shrink(b, ctrl, 0.18)
  let dir = (tip.at(0) - ctrl.at(0), tip.at(1) - ctrl.at(1))
  let len = calc.sqrt(calc.pow(dir.at(0) / 1pt, 2) + calc.pow(dir.at(1) / 1pt, 2))
  let u = (dir.at(0) / len, dir.at(1) / len)
  let back = (tip.at(0) - 7 * u.at(0), tip.at(1) - 7 * u.at(1))
  let side = (-u.at(1) * 3.5, u.at(0) * 3.5)
  place(polygon(fill: color, tip,
    (back.at(0) + side.at(0), back.at(1) + side.at(1)),
    (back.at(0) - side.at(0), back.at(1) - side.at(1))))
  let m = (0.5 * ctrl.at(0) + 0.25 * (a.at(0) + b.at(0)), 0.5 * ctrl.at(1) + 0.25 * (a.at(1) + b.at(1)))
  let off = shrink(m, c, -0.9)
  label(off, text(fill: color, size: 10pt, strong(str(n))))
}

#for p in pos { place(line(start: c, end: p, stroke: 3pt + luma(200))) }
#hop(1, 0, blue, 1)
#hop(2, 1, orange, 2)
#hop(0, 2, blue, 3)
#dot(c, luma(120))
#for p in pos { dot(p, white) }
#label((c.at(0) + 16pt, c.at(1) + 4pt), $γ_0$)
#label((pos.at(0).at(0) + 20pt, pos.at(0).at(1)), $γ_1$)
#label((pos.at(1).at(0) - 4pt, pos.at(1).at(1) + 16pt), $γ_2$)
#label((pos.at(2).at(0) + 4pt, pos.at(2).at(1) + 16pt), $γ_3$)

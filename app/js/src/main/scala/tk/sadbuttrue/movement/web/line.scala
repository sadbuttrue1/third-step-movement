package tk.sadbuttrue.movement.web

import japgolly.scalajs.react._
import japgolly.scalajs.react.vdom.svg.all._

//import japgolly.scalajs.react.vdom.svg.prefix_<^._
import paths.high.Stock
import tk.sadbuttrue.movement.web.colors._

/**
  * Created by true on 03/02/16.
  */
object Line {
  private val palette = mix(Color(130, 140, 210), Color(180, 205, 150))

  val LineChart = ReactComponentB[(Seq[Seq[(Double, Double)]], Seq[String])]("Stock chart")
    .render(points => {
      val stock = Stock[(Double, Double)](
        data = points.props._1,
        xaccessor = _._1,
        yaccessor = _._2,
        width = 420,
        height = 360,
        closed = true
      )
      val lines = stock.curves map { curve =>
        g(transform := "translate(0,0)",
          path(d := curve.line.path.print, fill := "none", stroke := string(palette(curve.index)))
        )
      }

      val legends = stock.curves.map { curve =>
        val translate = s"translate(0,${30 * curve.index})"
        val name = points.props._2(curve.index)
        g(transform := translate,
          rect(width := 20, height := 20, fill := string(palette(curve.index))),
          text(transform := "translate(30, 15)", fontSize := 12)(name)
        )
      }

      svg(width := 640, height := 480,
        lines,
        legends
      )
    })
    .build
}

object colors {

  case class Color(r: Double, g: Double, b: Double, alpha: Double = 1)

  def cut(x: Double) = x.floor min 255

  def multiply(factor: Double) = { c: Color =>
    Color(cut(factor * c.r), cut(factor * c.g), cut(factor * c.b), c.alpha)
  }

  def average(c1: Color, c2: Color) =
    Color(
      cut((c1.r + c2.r) / 2),
      cut((c1.g + c2.g) / 2),
      cut((c1.b + c2.b) / 2),
      (c1.alpha + c2.alpha / 2)
    )

  val lighten = multiply(1.3)
  val darken = multiply(0.7)

  def mix(c1: Color, c2: Color) = {
    val c3 = average(c1, c2)
    List(
      lighten(c1),
      c1,
      darken(c1),
      lighten(c3),
      c3,
      darken(c3),
      lighten(c2),
      c2,
      darken(c2)
    )
  }

  def transparent(c: Color, alpha: Double = 0.7) = c.copy(alpha = alpha)

  def string(c: Color) =
    if (c.alpha == 1) s"rgb(${c.r.floor},${c.g.floor},${c.b.floor})"
    else s"rgba(${c.r.floor},${c.g.floor},${c.b.floor},${c.alpha})"
}
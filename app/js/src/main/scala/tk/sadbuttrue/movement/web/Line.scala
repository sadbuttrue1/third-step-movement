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
      val data = points.props._1
      val names = points.props._2
      val xMax = data.map { line =>
        line.maxBy(_._1)
      }.maxBy(_._1)._1
      val xMin = data.map { line =>
        line.minBy(_._1)
      }.minBy(_._1)._1
      val yMax = data.map { line =>
        line.maxBy(_._2)
      }.maxBy(_._2)._2
      val yMin = data.map { line =>
        line.minBy(_._2)
      }.minBy(_._2)._2

      val stock = Stock[(Double, Double)](
        data = data,
        xaccessor = _._1,
        yaccessor = _._2,
        width = 500,
        height = 460,
        closed = true
      )

      val lines = stock.curves map { curve =>
        g(transform := "translate(20,0)",
          path(d := curve.line.path.print, fill := "none", stroke := string(palette(curve.index)))
        )
      }

      val xscale = stock.xscale
      val yscale = stock.yscale
      val axes = g(transform := "translate(20,0)",
        line(x1 := xscale(xMin), y1 := yscale(yMin) + 5, x2 := xscale(xMin), y2 := yscale(yMax) - 5, stroke := "#333333"),
        line(x1 := xscale(xMin) - 5, y1 := yscale(yMin), x2 := xscale(xMax) + 5, y2 := yscale(yMin), stroke := "#333333")
      )

      val xStep = (xMax - xMin) / 10.0
      val yStep = (yMax - yMin) / 10.0
      val grid = g(transform := "translate(20,0)",
        for (i <- 1 to 10) yield {
          line(x1 := xscale(xMin + i * xStep), y1 := yscale(yMin) + 5, x2 := xscale(xMin + i * xStep), y2 := yscale(yMax) - 5, stroke := "black", strokeWidth := 0.1)
        },
        for (i <- 1 to 10) yield {
          line(x1 := xscale(xMin) - 5, y1 := yscale(yMin + i * yStep), x2 := xscale(xMax) + 5, y2 := yscale(yMin + i * yStep), stroke := "black", strokeWidth := 0.1)
        }
      )

      val xNumbers = g(
        for (i <- 1 to 10) yield {
          text(transform := s"translate(${xscale(xMin + i * xStep)}, ${yscale(yMin) + 10})", fontSize := 8)("%.2f" format xMin + i * xStep)
        }
      )

      val yNumbers = g(
        for (i <- 0 to 10) yield {
          text(transform := s"translate(${xscale(xMin) + 5}, ${yscale(yMin + i * yStep)}) rotate(90)", fontSize := 8)("%.2f" format yMin + i * yStep)
        }
      )

      val legends = stock.curves.map { curve =>
        val translate = s"translate(520, ${30 * curve.index})"
        val name = names(curve.index)
        g(transform := translate,
          rect(width := 20, height := 20, fill := string(palette(curve.index))),
          text(transform := "translate(20, 15)", fontSize := 12)(name)
        )
      }

      svg(width := 640, height := 480,
        lines,
        legends,
        axes,
        grid,
        xNumbers,
        yNumbers
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
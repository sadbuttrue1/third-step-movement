package tk.sadbuttrue.movement.web

import japgolly.scalajs.react._
import org.scalajs.dom._
import org.scalajs.dom.ext.Ajax
import org.scalajs.dom.raw.{HTMLParagraphElement, HTMLInputElement}

import scala.concurrent.Future
import scala.scalajs.js
import japgolly.scalajs.react.vdom.prefix_<^._

import tk.sadbuttrue.movement.util.model.Task

import scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
import scala.scalajs.js.annotation.JSExport

import autowire._

object Ajaxer extends autowire.Client[String, upickle.default.Reader, upickle.default.Writer] {
  override def doCall(req: Request) = {
    ext.Ajax.post(
      url = "/ajax/" + req.path.mkString("/"),
      data = upickle.default.write(req.args)
    ).map(_.responseText)
  }

  def read[Result: upickle.default.Reader](p: String) = upickle.default.read[Result](p)

  def write[Result: upickle.default.Writer](r: Result) = upickle.default.write(r)
}

object Client extends js.JSApp {
  val output = Ref[HTMLParagraphElement]("output")
  val jsonFile = Ref[HTMLInputElement]("jsonFile")

  implicit def futToCallback(fut: Future[Callback]): Callback = Callback(fut.foreach(_.runNow()))

  case class State(
                    selected: Boolean,
                    v: Option[Seq[Seq[(Double, Double)]]],
                    a: Option[Seq[Seq[(Double, Double)]]],
                    w_d: Option[Seq[Seq[(Double, Double)]]],
                    q_d: Option[Seq[Seq[(Double, Double)]]]
                  )

  class Backend($: BackendScope[Unit, State]) {
    def selected(e: ReactEventI) = {
      $.modState(_.copy(selected = true))
    }

    def send = {
      val reader = new FileReader
      val file = jsonFile($).get.files(0)
      reader.readAsText(file)
      reader.onload = (ev: Event) => {
        val task = upickle.default.read[Task](reader.result.asInstanceOf[String])
        Ajaxer[Api].calculate(task).
          call().map { result =>
          val v = Some(
            List(
              result.t.zip(result.v(0)).toList,
              result.t.zip(result.v(1)).toList,
              result.t.zip(result.v(2)).toList
            )
          )
          val a = Some(
            List(
              result.t.zip(result.a(0)).toList,
              result.t.zip(result.a(1)).toList,
              result.t.zip(result.a(2)).toList
            )
          )
          val w_d = Some(
            List(
              result.t.zip(result.w_d(0)).toList,
              result.t.zip(result.w_d(1)).toList,
              result.t.zip(result.w_d(2)).toList
            )
          )
          val q_d = Some(
            List(
              result.t.zip(result.quat_d(0)).toList,
              result.t.zip(result.quat_d(1)).toList,
              result.t.zip(result.quat_d(2)).toList,
              result.t.zip(result.quat_d(3)).toList
            )
          )
          $.modState(_.copy(
            v = v,
            a = a,
            w_d = w_d,
            q_d = q_d
          ))
        }
      }.runNow()
      $.modState(_.copy(selected = false))
    }

    def render(s: State) = {
      <.div(
        <.h1("choose json-file with task"),
        <.p(<.input(^.`type` := "file", ^.accept := "application/json", ^.multiple := false, ^.onChange ==> selected, ^.ref := jsonFile)),
        <.p(^.ref := output),
        <.p(<.input(^.`type` := "button", ^.value := "send", ^.onClick --> send, ^.disabled := !s.selected)),
        s.v.fold(<.div)(data => <.div(<.p("v:"), Line.LineChart(data))),
        s.a.fold(<.div)(data => <.div(<.p("a:"), Line.LineChart(data))),
        s.w_d.fold(<.div)(data => <.div(<.p("w_d:"), Line.LineChart(data))),
        s.q_d.fold(<.div)(data => <.div(<.p("q_d:"), Line.LineChart(data)))
      )

    }
  }

  val MovementApp = ReactComponentB[Unit]("MovementApp")
    .initialState(State(false, None, None, None, None))
    .renderBackend[Backend]
    .buildU

  @JSExport
  override def main() = ReactDOM.render(MovementApp(), document.getElementById("contents"))
}
package tk.sadbuttrue.movement.web

import japgolly.scalajs.react.{BackendScope, ReactDOM, ReactComponentB, ReactEventI, Ref}
import org.scalajs.dom._
import org.scalajs.dom.raw.{HTMLParagraphElement, HTMLInputElement}

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
  val sendButton = Ref[HTMLInputElement]("sendButton")

  class Backend($: BackendScope[Unit, Unit]) {
    def selected(e: ReactEventI) = {
      sendButton($).get.disabled = false
      $.setState(e.target.value)
    }

    def send = {
      val reader = new FileReader
      val file = jsonFile($).get.files(0)
      reader.readAsText(file)
      reader.onload = (ev: Event) => {
        val task = upickle.default.read[Task](reader.result.asInstanceOf[String])
        output($).get.textContent = task.toString
        Ajaxer[Api].calculate(task).
          call().foreach { result =>
          output($).get.textContent = result.toString
        }
      }
      $.state
    }

    def render =
      <.div(
        <.h1("choose json-file with task"),
        <.p(<.input(^.`type` := "file", ^.accept := "application/json", ^.multiple := false, ^.onChange ==> selected, ^.ref := jsonFile)),
        <.p(^.ref := output),
        <.p(<.input(^.`type` := "button", ^.value := "send", ^.onClick --> send, ^.disabled := true, ^.ref := sendButton))
      )
  }

  val MovementApp = ReactComponentB[Unit]("MovementApp")
    .renderBackend[Backend]
    .buildU

  @JSExport
  override def main() = ReactDOM.render(MovementApp(), document.getElementById("contents"))
}

///**
//  * Created by true on 31/01/16.
//  */
//
//import autowire._
//import japgolly.scalajs.react.ReactComponentB
//import org.scalajs.dom
//import org.scalajs.dom.html
//import paths.high.Stock
//import tk.sadbuttrue.movement.util.model.Task
//
//import scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
//import scala.scalajs.js.annotation.JSExport
//import scalatags.JsDom.all._
//

//
//@JSExport
//object Client extends {
//  @JSExport
//  def main(container: html.Div) = {
//    val inputBox = input.render
//    val output = span.render
//    val jsonFile = input(`type` := "file", title := "json file", accept := "application/json", multiple := false).render
//    val sendButton = input(`type` := "button", value := "send").render
//    val clearButton = input(`type` := "button", value := "clear").render
//    var task: Task = null
//    def update() = Ajaxer[Api].calculate(task).
//      call().foreach { data =>
//      output.innerHTML = ""
//      output.textContent = data.toString
//      output.render
//    }
//    def show() = {
//      val file = jsonFile.files.item(0)
//      val reader = new dom.FileReader
//      reader.readAsText(file)
//      reader.onload = (e: dom.Event) => {
//        task = upickle.default.read[Task](reader.result.asInstanceOf[String])
//      }
//    }
//    def calculate() = Ajaxer[Api].calculate(task).
//      call().foreach { result =>
//      output.textContent = result.toString
//      output.render
//    }
//    inputBox.onkeyup = (e: dom.Event) => update()
//    jsonFile.onchange = (e: dom.Event) => show()
//    sendButton.onclick = (e: dom.Event) => calculate()
//    container.appendChild(
//      div(
//        h1("choose json-file with task"),
//        p(jsonFile),
//        p(sendButton, clearButton),
//        output
//      ).render
//    )
//  }
//}

//object line {
//  import japgolly.scalajs.react._
//  import japgolly.scalajs.react.vdom.svg.all._
//  import paths.high.Stock
//
//  case class Event(x: Double, y: Double)
//
//
//  val LineChart = ReactComponentB[Seq[Seq[Event]]]("Stock chart")
//    .render(events => {
//      val stock = Stock[Event](
//        data = events,
//        xaccessor = _.x,
//        yaccessor = _.y,
//        width = 420,
//        height = 360,
//        closed = true
//      )
//      val lines = stock.curves map { curve =>
//        g(transform := "translate(50,0)",
//          path(d := curve.area.path.print, fill := "none", stroke := "none"),
//          path(d := curve.line.path.print, fill := "none", stroke := "none")
//        )
//      }
//
//      svg(width := 480, height := 400,
//        lines
//      )
//    })
//    .build
//}
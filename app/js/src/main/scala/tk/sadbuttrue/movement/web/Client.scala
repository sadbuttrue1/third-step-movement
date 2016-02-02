package tk.sadbuttrue.movement.web

/**
  * Created by true on 31/01/16.
  */

import tk.sadbuttrue.movement.util.model.{DoubleWithError, Task}

import scala.io.Source
import scalatags.JsDom.all._
import org.scalajs.dom
import dom.html
import scalajs.js.annotation.JSExport
import scalajs.concurrent.JSExecutionContext.Implicits.queue
import autowire._

object Ajaxer extends autowire.Client[String, upickle.default.Reader, upickle.default.Writer] {
  override def doCall(req: Request) = {
    dom.ext.Ajax.post(
      url = "/ajax/" + req.path.mkString("/"),
      data = upickle.default.write(req.args)
    ).map(_.responseText)
  }

  def read[Result: upickle.default.Reader](p: String) = upickle.default.read[Result](p)

  def write[Result: upickle.default.Writer](r: Result) = upickle.default.write(r)
}

@JSExport
object Client extends {
  @JSExport
  def main(container: html.Div) = {
    val inputBox = input.render
    val output = span.render
    val jsonFile = input(`type` := "file", title := "json file", accept := "application/json", multiple := false).render
    val sendButton = input(`type` := "button", value := "send").render
    val clearButton = input(`type` := "button", value := "clear").render
    var task: Task = null
    def update() = Ajaxer[Api].calculate(task).
      call().foreach { data =>
      output.innerHTML = ""
      output.textContent = data.toString
      output.render
    }
    def show() = {
      val file = jsonFile.files.item(0)
      val reader = new dom.FileReader
      reader.readAsText(file)
      reader.onload = (e: dom.Event) => {
        task = upickle.default.read[Task](reader.result.asInstanceOf[String])
      }
    }
    def calculate() = Ajaxer[Api].calculate(task).
      call().foreach { result =>
      output.textContent = result.toString
      output.render
    }
    inputBox.onkeyup = (e: dom.Event) => update()
    jsonFile.onchange = (e: dom.Event) => show()
    sendButton.onclick = (e: dom.Event) => calculate()
    container.appendChild(
      div(
        h1("choose json-file with task"),
        p(jsonFile),
        p(sendButton, clearButton),
        output
      ).render
    )
  }
}
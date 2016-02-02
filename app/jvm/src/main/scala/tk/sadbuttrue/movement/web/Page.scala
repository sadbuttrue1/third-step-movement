package tk.sadbuttrue.movement.web

import scalatags.Text.all._

object Page{
  val boot =
    "tk.sadbuttrue.movement.web.Client().main(document.getElementById('contents'))"
  val skeleton =
    html(
      head(
        script(src:="/third-step-movement-opt.js"),
        link(
          rel:="stylesheet",
          href:="https://cdnjs.cloudflare.com/ajax/libs/pure/0.5.0/pure-min.css"
        )
      ),
      body(
        onload:=boot,
        div(id:="contents")
      )
    )
}
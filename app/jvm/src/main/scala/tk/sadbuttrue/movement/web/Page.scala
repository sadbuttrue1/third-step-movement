package tk.sadbuttrue.movement.web

import scalatags.Text.all._

object Page {
  val boot =
    "tk.sadbuttrue.movement.web.Client().main(document.getElementById('contents'))"
  val skeleton =
    html(
      head(
        meta(httpEquiv := "Content-Type", content := "text/html; charset=UTF-8"),
        link(
          rel := "stylesheet",
          href := "https://cdnjs.cloudflare.com/ajax/libs/pure/0.5.0/pure-min.css"
        )
      ),
      body(
//        onload := boot,
        div(id := "contents"),
        script(`type` := "text/javascript", src := "/third-step-movement-jsdeps.js"),
        script(`type` := "text/javascript", src := "/third-step-movement-fastopt.js"),
        script(`type` := "text/javascript", src := "/third-step-movement-launcher.js")
      )
    )
}
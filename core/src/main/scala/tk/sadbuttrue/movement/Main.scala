package tk.sadbuttrue.movement

import tk.sadbuttrue.movement.math.Functions
import tk.sadbuttrue.movement.util.model.Task

/**
  * Created by true on 25/01/16.
  */
object Main extends App {
  val tmp = Task("sample.json")
  println(tmp)
  val P = Functions.p(tmp)
  println(P(47.0))
}

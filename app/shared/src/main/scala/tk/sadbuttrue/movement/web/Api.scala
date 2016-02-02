package tk.sadbuttrue.movement.web

import tk.sadbuttrue.movement.util.model.{Result, Task}

/**
  * Created by true on 31/01/16.
  */
trait Api {
  def calculate(task: Task): Result
}

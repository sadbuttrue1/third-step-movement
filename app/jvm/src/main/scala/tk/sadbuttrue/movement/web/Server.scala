package tk.sadbuttrue.movement.web

import akka.actor.ActorSystem
import spray.http.{HttpEntity, MediaTypes}
import spray.routing.SimpleRoutingApp
import tk.sadbuttrue.movement.math.Solver
import tk.sadbuttrue.movement.util.model.{ServerTask1, ServerTask, Result, Task}

import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by true on 31/01/16.
  */
object Router extends autowire.Server[String, upickle.default.Reader, upickle.default.Writer] {
  def read[Result: upickle.default.Reader](p: String) = upickle.default.read[Result](p)

  def write[Result: upickle.default.Writer](r: Result) = upickle.default.write(r)
}

object Server extends SimpleRoutingApp with Api {
  def main(args: Array[String]): Unit = {
    implicit val system = ActorSystem()
    startServer("localhost", port = 8080) {
      get {
        pathSingleSlash {
          complete {
            HttpEntity(
              MediaTypes.`text/html`,
              Page.skeleton.render
            )
          }
        } ~
          getFromResourceDirectory("")
      } ~
        post {
          path("ajax" / Segments) { s =>
            extract(_.request.entity.asString) { e =>
              complete {
                Router.route[Api](Server)(
                  autowire.Core.Request(
                    s,
                    upickle.default.read[Map[String, String]](e)
                  )
                )
              }
            }
          }
        }
    }
  }

  def calculate(task: Task): Result = {
    Solver.solve(ServerTask1(task))
  }
}
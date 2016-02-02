package tk.sadbuttrue.movement.util.model

//import java.nio.charset.StandardCharsets
//import java.nio.file.{Files, Paths}
//
//import org.json4s.{Extraction, NoTypeHints, CustomKeySerializer}
//import org.json4s.jackson.Serialization
//import org.json4s._
//import org.json4s.jackson.JsonMethods._
//
//import scala.io.Source

/**
  * Created by true on 25/01/16.
  */
case class Task(m: DoubleWithError, T: Double,
                L: Double, l: Double, R: Double,
                p: Map[Double, DoubleWithError],
                r_c: List[DoubleWithError],
                J: List[DoubleWithError])

//object Task {
//  val DoubleSerializer = new CustomKeySerializer[Double](
//    format => ( {
//      case s: String => s.toDouble
//    }, {
//      case k: Double => k.toString
//    }
//      ))
//
//  implicit val serializationFormats = {
//    Serialization.formats(NoTypeHints) + DoubleSerializer
//  }
//
//  def apply(jsonFilename: String): Task = {
//    val fileContents = Source.fromFile(jsonFilename, "UTF-8").getLines.mkString
//    parse(fileContents).extract[Task]
//  }
//
//  def saveJson(task: Task, jsonFilename: String): Unit = {
//    val txt = pretty(render(Extraction.decompose(task)))
//    Files.write(Paths.get(jsonFilename), txt.getBytes(StandardCharsets.UTF_8))
//  }
//}

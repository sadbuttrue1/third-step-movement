package tk.sadbuttrue.movement.util.model

/**
  * Created by true on 18/02/16.
  */
case class ServerTask(m: Double, T: Double,
                      L: Double, l: Double, D: Double,
                      p: Map[Double, Double],
                      r_c: List[Double],
                      J: List[Double]) {

  override def toString: String = {
    var builder = new StringBuilder
    builder ++= s"m = ${this.m}\n"
    builder ++= s"T = ${this.T}\n"
    builder ++= s"L = ${this.L}\n"
    builder ++= s"l = ${this.l}\n"
    builder ++= s"D = ${this.D}\n"
    builder ++= s"p = ${this.p}\n"
    builder ++= s"r_c = ${this.r_c}\n"
    builder ++= s"J = ${this.J}\n"

    builder.toString
  }
}
package tk.sadbuttrue.movement.util.model

import DoubleWithErrorRandomHelper.doubleWithErrorToDouble
import org.apache.commons.math3.distribution.NormalDistribution

/**
  * Created by true on 18/02/16.
  */
object ServerTask1 {
  private val random = new NormalDistribution(0, 1.0 / 3.0)

  def apply(task: Task): ServerTask = {
    val m: Double = task.m
    val T: Double = task.T
    val L: Double = task.L
    val l: Double = task.l
    val D: Double = task.D
    val p: Map[Double, Double] = task.p.map(p => p._1 -> p._2.toDouble)
    val r_c: List[Double] = task.r_c.map(r_c => r_c.toDouble)
    val k = random.sample
    val J: List[Double] = List(
      task.J(0).value + k * task.J(0).error,
      task.J(1).value + k * task.J(1).error,
      task.J(2).value + k * task.J(2).error
    )

    ServerTask(m, T, L, l, D, p, r_c, J)
  }
}
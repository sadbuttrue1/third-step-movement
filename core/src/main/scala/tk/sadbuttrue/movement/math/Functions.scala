package tk.sadbuttrue.movement.math

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{sin, toRadians, cos}
import breeze.numerics.constants.Pi
import tk.sadbuttrue.movement.util.model.Task

import scala.annotation.switch

/**
  * Created by true on 27/01/16.
  */
object Functions {
  def p(task: Task) = (t: Double) => {
    val max = task.p.maxBy(_._1)._1
    val min = task.p.minBy(_._1)._1
    val range = (t: @switch) match {
      case t if t == max => (task.p.filterKeys(k => k < max).maxBy(_._1), task.p.filterKeys(k => k == max).minBy(_._1))
      case t if t > max => (max -> task.p(max), max -> task.p(max))
      case t if t == min => (task.p.filterKeys(k => k == min).maxBy(_._1), task.p.filterKeys(k => k > min).minBy(_._1))
      case _ => (task.p.filterKeys(k => k <= t).maxBy(_._1), task.p.filterKeys(k => k > t).minBy(_._1))
    }
    val b = DenseVector[Double](range._1._2, range._2._2)
    val a = DenseMatrix.zeros[Double](2, 2)
    a(0, 0) = range._1._1
    a(0, 1) = 1
    a(1, 0) = range._2._1
    a(1, 1) = 1
    val x = a \ b
    x(0) * t + x(1)
  }

  def n_p(): DenseVector[Double] = {
    val alpha = toRadians(20)
    val beta = toRadians(41)
    val gamma = toRadians(49)
    DenseVector[Double](-cos(alpha), -cos(beta) * sin(alpha), -cos(gamma) * sin(alpha))
  }

  def rho(task: Task): DenseVector[Double] = {
    val x_c: Double = task.r_c(0)
    val y_c: Double = task.r_c(1)
    val z_c: Double = task.r_c(2)
    DenseVector[Double](task.L - task.l - x_c, cos(Pi / 4.0) * task.R - y_c, sin(Pi / 4.0) * task.R - z_c)
  }
}

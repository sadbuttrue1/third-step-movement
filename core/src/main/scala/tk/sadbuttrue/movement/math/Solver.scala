package tk.sadbuttrue.movement.math

import breeze.linalg.DenseVector
import breeze.linalg.cross
import org.apache.commons.math3.complex.Quaternion
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations
import tk.sadbuttrue.movement.util.model.Task
import tk.sadbuttrue.movement.math.Functions._

/**
  * Created by true on 27/01/16.
  */
object Solver {

}


class MovementODE(val task: Task) extends FirstOrderDifferentialEquations {
  private val random = new NormalDistribution

  override def getDimension: Int = 13

  override def computeDerivatives(t: Double, y: Array[Double], yDot: Array[Double]): Unit = {
    val r = DenseVector[Double](y(0), y(1), y(2))
    val v = DenseVector[Double](y(3), y(4), y(5))
    val w = DenseVector[Double](y(6), y(7), y(8))
    val quat = new Quaternion(y(9), y(10), y(11), y(12))
    val P = p(task)
    val P_proj: DenseVector[Double] = n_p * P(t)
    val Rho = rho(task)
    val M = cross(Rho, P_proj)
    val F = DenseVector[Double](((quat multiply new Quaternion(P_proj.toArray)) multiply (quat getConjugate)) getVectorPart)
    val k = random.sample
    val J = DenseVector[Double](task.J(0).value + k * task.J(0).error, task.J(1).value + k * task.J(1).error, task.J(2).value + k * task.J(2).error)

    val m: Double = task.m

    val a = F / m
  }
}
package tk.sadbuttrue.movement.math

import breeze.linalg.{DenseMatrix, DenseVector, cross}
import org.apache.commons.math3.complex.Quaternion
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator
import org.apache.commons.math3.ode.sampling.{StepInterpolator, StepHandler}
import org.apache.commons.math3.ode.{FirstOrderIntegrator, FirstOrderDifferentialEquations}
import tk.sadbuttrue.movement.util.model.{Result, Task}

import tk.sadbuttrue.movement.util.model.DoubleWithErrorRandomHelper.doubleWithErrorToDouble
import tk.sadbuttrue.movement.math.Functions._
import tk.sadbuttrue.movement.math.QuaternionHelper.rotate

import breeze.plot._

/**
  * Created by true on 27/01/16.
  */
object Solver {
  def solve(task: Task): Result = {
    val integrator: FirstOrderIntegrator = new DormandPrince54Integrator(1.0e-8, task.T, 1.0e-10, 1.0e-10)
    val ode: FirstOrderDifferentialEquations = new MovementODE(task)
    val y0: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    var result: Array[Double] = Array()

    val stepHandler: StepHandler = new StepHandler {
      override def init(t0: Double, y0: Array[Double], t: Double): Unit = {}

      override def handleStep(interpolator: StepInterpolator, isLast: Boolean): Unit = {
        val y = interpolator.getInterpolatedState
        val t = interpolator.getCurrentTime
        result = Array.concat(result, y :+ t)
      }
    }

    integrator.addStepHandler(stepHandler)
    integrator.integrate(ode, 0.0, y0, task.T, y0)

    println("DONE")

    val matr = new DenseMatrix[Double](14, result.length / 14, result)

    val t = matr(13, ::).inner.toArray
    val v = List(matr(0, ::).inner.toArray, matr(1, ::).inner.toArray, matr(2, ::).inner.toArray)
    val a = List(matr(3, ::).inner.toArray, matr(4, ::).inner.toArray, matr(5, ::).inner.toArray)
    val w_d = List(matr(6, ::).inner.toArray, matr(7, ::).inner.toArray, matr(8, ::).inner.toArray)
    val quat_d = List(matr(9, ::).inner.toArray, matr(10, ::).inner.toArray, matr(11, ::).inner.toArray, matr(12, ::).inner.toArray)
    val f = Figure()
    f.visible = false
    val p = f.subplot(0)
    p += plot(x = matr(13, ::).inner, y = matr(6, ::).inner, name = "w_x")
    p += plot(x = matr(13, ::).inner, y = matr(7, ::).inner, name = "w_y")
    p += plot(x = matr(13, ::).inner, y = matr(8, ::).inner, name = "w_z")
    p.legend = true
    f.saveas("w.png")

    Result(t, v, a, w_d, quat_d)
  }
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
    val np = n_p()
    val P_proj = np :* P(t)
    val Rho = rho(task)
    val M = cross(Rho, P_proj)
    val F = rotate(P_proj, quat)
    val k = random.sample
    val J = DenseVector[Double](
      task.J(0).value + k * task.J(0).error,
      task.J(1).value + k * task.J(1).error,
      task.J(2).value + k * task.J(2).error
    )

    val m: Double = task.m

    val a: DenseVector[Double] = F :/ m

    val w_d = M :/ J
    w_d(0) += (J(1) - J(2)) * w(1) * w(2) / J(0)
    w_d(1) += (J(2) - J(0)) * w(2) * w(0) / J(1)
    w_d(2) += (J(0) - J(1)) * w(0) * w(1) / J(2)

    val quat_d = new Quaternion(
      (-w(0) * quat.getQ1 - w(1) * quat.getQ2 - w(2) * quat.getQ3) / 2.0,
      (w(0) * quat.getQ0 + w(2) * quat.getQ2 - w(1) * quat.getQ3) / 2.0,
      (w(1) * quat.getQ0 - w(2) * quat.getQ1 + w(0) * quat.getQ3) / 2.0,
      (w(2) * quat.getQ0 + w(1) * quat.getQ1 - w(0) * quat.getQ2) / 2.0
    )

    yDot(0) = v(0)
    yDot(1) = v(1)
    yDot(2) = v(2)
    yDot(3) = a(0)
    yDot(4) = a(1)
    yDot(5) = a(2)
    yDot(6) = w_d(0)
    yDot(7) = w_d(1)
    yDot(8) = w_d(2)
    yDot(9) = quat_d.getQ0
    yDot(10) = quat_d.getQ1
    yDot(11) = quat_d.getQ2
    yDot(12) = quat_d.getQ3
  }
}
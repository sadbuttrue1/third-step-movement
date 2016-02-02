package tk.sadbuttrue.movement.math

import breeze.linalg.DenseVector
import org.apache.commons.math3.complex.Quaternion

/**
  * Created by true on 31/01/16.
  */
object QuaternionHelper {
  def rotate(vector: DenseVector[Double], quaternion: Quaternion): DenseVector[Double] = {
    DenseVector[Double](((quaternion multiply new Quaternion(vector.toArray)) multiply (quaternion getConjugate)) getVectorPart)
  }
}

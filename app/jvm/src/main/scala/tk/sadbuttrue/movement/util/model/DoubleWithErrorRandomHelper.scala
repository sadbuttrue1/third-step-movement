package tk.sadbuttrue.movement.util.model

import org.apache.commons.math3.distribution.NormalDistribution

/**
  * Created by true on 31/01/16.
  */
object DoubleWithErrorRandomHelper {
  private val random = new NormalDistribution(0, 1.0 / 3.0)

  implicit def doubleWithErrorToDouble(d: DoubleWithError): Double = {
    d.value + d.error * random.sample()
  }
}

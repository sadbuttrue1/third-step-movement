package tk.sadbuttrue.movement.util.model

//import org.apache.commons.math3.distribution.NormalDistribution

/**
  * Created by true on 25/01/16.
  */
case class DoubleWithError(value: Double, error: Double) {
  //  private val random = new NormalDistribution

  override def toString(): String = {
    s"$value ± $error"
  }
}

object DoubleWithError {
  implicit def dobleWithErrorToString(d: DoubleWithError): String = {
    s"${d.value} ± ${d.error}"
  }

  implicit def stringToDoubleWithError(s: String): DoubleWithError = {
    val subs = s.split("±")
    DoubleWithError(subs(0).toDouble, subs(1).toDouble)
  }

  implicit def doubleWithErrorToDouble(d: DoubleWithError): Double = {
    d.value //+ d.error * d.random.sample()
  }
}

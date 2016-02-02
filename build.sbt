val app = crossProject.settings(
  unmanagedSourceDirectories in Compile +=
    baseDirectory.value / "shared" / "main" / "scala",
  libraryDependencies ++= Seq(
    "com.lihaoyi" %%% "scalatags" % "0.5.4",
    "com.lihaoyi" %%% "upickle" % "0.3.7",
    "com.lihaoyi" %%% "autowire" % "0.2.5"
  ),
  organization := "tk.sadbuttrue",
  name := "third-step-movement",
  version := "0.1",
  scalaVersion := "2.11.7"
).enablePlugins(JavaServerAppPackaging)
  .jsSettings(
    libraryDependencies ++= Seq(
      "org.scala-js" %%% "scalajs-dom" % "0.8.0",
      "com.github.japgolly.scalajs-react" %%% "core" % "0.8.1",
      "eu.unicredit" %%% "paths-scala-js" % "0.4.0"
    ),
    jsDependencies ++= Seq(
      "org.webjars" % "react" % "0.12.1" / "react-with-addons.js" commonJSName "React"
    )
  ).jvmSettings(
  libraryDependencies ++= Seq(
    "io.spray" %% "spray-can" % "1.3.3",
    "io.spray" %% "spray-routing" % "1.3.3",
    "com.typesafe.akka" %% "akka-actor" % "2.3.9",
    "org.scalanlp" %% "breeze" % "0.11.2",
    "org.scalanlp" %% "breeze-natives" % "0.11.2",
    "org.apache.commons" % "commons-math3" % "3.6",
    "org.scalanlp" %% "breeze-viz" % "0.11.2"
  )
)

lazy val appJS = app.js
lazy val appJVM = app.jvm.settings(
  (resources in Compile) += (fullOptJS in(appJS, Compile)).value.data
)



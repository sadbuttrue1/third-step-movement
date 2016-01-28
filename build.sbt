import sbt.Keys._

lazy val commonSettings = Seq(
  organization := "tk.sadbuttrue",
  version := "0.1",
  scalaVersion := "2.11.7"
)

lazy val root = (project in file(".")).
  settings(name := "third-step-movement").
  aggregate(core, web).settings(run := {
  (run in core in Runtime).evaluated;
  (run in web in Runtime).evaluated
})

val appName = "third-step-movement"

lazy val api = project.settings(commonSettings: _*).
  settings(name := appName + "-api").
  settings(libraryDependencies += "org.json4s" %% "json4s-jackson" % "3.3.0").
  settings(libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6")

lazy val core = project.settings(commonSettings: _*).
  settings(name := appName + "-core").
  settings(libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6").
  settings(libraryDependencies += "org.scalanlp" %% "breeze" % "0.11.2").
  settings(libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.11.2").
  dependsOn(api)

lazy val web = project.settings(commonSettings: _*).
  settings(name := appName + "-web").enablePlugins(ScalaJSPlugin)


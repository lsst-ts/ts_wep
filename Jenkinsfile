#!/usr/bin/env groovy

pipeline {

    agent {
        // Use the docker to assign the Python version.
        // Use the label to assign the node to run the test.
        // It is recommended by SQUARE team do not add the label to let the
        // system decide.
        docker {
          image 'lsstts/develop-env:develop'
          args "--entrypoint=''"
          alwaysPull true
          // Specify the nodes that have the git-lfs tool installed.
          // If run on nodes without git-lfs the tests will fail.
          label 'Node3_4CPU  || CSC_Conda_Node'
        }
    }

    options {
      disableConcurrentBuilds()
      skipDefaultCheckout()
    }

    triggers {
        cron(env.BRANCH_NAME == 'develop' ? '0 4 * * *' : '')
    }

    environment {
        // Position of LSST stack directory
        LSST_STACK = "/opt/lsst/software/stack"
        // Pipeline Sims Version
        STACK_VERSION = "current"
        // XML report path
        XML_REPORT = "jenkinsReport/report.xml"
        // Module name used in the pytest coverage analysis
        MODULE_NAME = "lsst.ts.wep"
        // PlantUML url
        PLANTUML_URL = "https://github.com/plantuml/plantuml/releases/download/v1.2021.13/plantuml-1.2021.13.jar"
        // Authority to publish the document online
        user_ci = credentials('lsst-io')
        LTD_USERNAME = "${user_ci_USR}"
        LTD_PASSWORD = "${user_ci_PSW}"
        DOCUMENT_NAME = "ts-wep"
    }

    stages {

        stage('Cloning Repos') {
            steps {
                dir(env.WORKSPACE + '/ts_wep') {
                    script {
                        if (env.BRANCH_NAME == 'develop' || env.BRANCH_NAME == 'main') {
                            SCM_BRANCH = env.BRANCH_NAME
                        }
                        else {
                            SCM_BRANCH = env.CHANGE_BRANCH
                        }
                    }
                    checkout scm
                }
            }
        }

        stage ('Building the Dependencies') {
            steps {
                // When using the docker container, we need to change
                // the HOME path to WORKSPACE to have the authority
                // to install the packages.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source ${env.LSST_STACK}/loadLSST.bash
                        cd ts_wep/
                        git lfs install || echo  git lfs install FAILED
                        git lfs fetch --all || echo git lfs fetch FAILED
                        git lfs checkout || echo git lfs checkout FAILED
                        setup -k -r .
                        scons python
                    """
                }
            }
        }

        stage('Unit Tests and Coverage Analysis') {
            steps {
                // Pytest needs to export the junit report.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source ${env.LSST_STACK}/loadLSST.bash
                        cd ts_wep/
                        setup -k -r .
                        pytest --cov-report html --cov=${env.MODULE_NAME} --junitxml=${env.WORKSPACE}/${env.XML_REPORT}
                    """
                }
            }
        }
    }

    post {
        always {
            // The path of xml needed by JUnit is relative to
            // the workspace.
            junit "${env.XML_REPORT}"

            // Publish the HTML report
            publishHTML (target: [
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'ts_wep/htmlcov',
                reportFiles: 'index.html',
                reportName: "Coverage Report"
            ])

            script{
              withEnv(["HOME=${env.WORKSPACE}"]) {
                def RESULT = sh returnStatus: true, script: """
                  source ${env.LSST_STACK}/loadLSST.bash

                  curl -L ${env.PLANTUML_URL} -o plantuml.jar

                  pip install sphinxcontrib-plantuml

                  cd ts_wep
                  setup -k -r .

                  package-docs build

                  pip install ltd-conveyor

                  ltd upload --product ${env.DOCUMENT_NAME} --git-ref ${env.BRANCH_NAME} --dir doc/_build/html
                    """

                if ( RESULT != 0 ) {
                    unstable("Failed to push documentation.")
                }
              }
            }
        }
        regression {
            script {
                slackSend(color: "danger", message: "${JOB_NAME} has suffered a regression ${BUILD_URL}", channel: "#aos-builds")
            }

        }
        fixed {
            script {
                slackSend(color: "good", message: "${JOB_NAME} has been fixed ${BUILD_URL}", channel: "#aos-builds")
            }
        }
        cleanup {
            // clean up the workspace
            deleteDir()
        }
    }
}

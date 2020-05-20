#!/usr/bin/env groovy

pipeline {

    agent {
        // Use the docker to assign the Python version.
        // Use the label to assign the node to run the test.
        // It is recommended by SQUARE team do not add the label to let the
        // system decide.
        docker {
            image 'lsstts/aos:w_2020_15'
        }
    }

    triggers {
        pollSCM('H * * * *')
    }

    environment {
        // Development tool set
        DEV_TOOL="/opt/rh/devtoolset-8/enable"
        // Position of LSST stack directory
        LSST_STACK="/opt/lsst/software/stack"
        // Pipeline Sims Version
        SIMS_VERSION="sims_w_2020_15"
        // XML report path
        XML_REPORT="jenkinsReport/report.xml"
        // Module name used in the pytest coverage analysis
        MODULE_NAME="lsst.ts.wep"
    }

    stages {
        stage ('Install Requirements') {
            steps {
                // When using the docker container, we need to change
                // the HOME path to WORKSPACE to have the authority
                // to install the packages.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source ${env.DEV_TOOL}
                        source ${env.LSST_STACK}/loadLSST.bash
                        git clone --branch master https://github.com/lsst-dm/phosim_utils.git
                        cd phosim_utils/
                        git checkout 8744592
                        setup -k -r . -t ${env.SIMS_VERSION}
                        scons
                        cd ..
                        setup -k -r .
                        python builder/setup.py build_ext --build-lib python/lsst/ts/wep/cwfs/lib
                    """
                }
            }
        }

        stage('Unit Tests and Coverage Analysis') {
            steps {
                // Direct the HOME to WORKSPACE for pip to get the
                // installed library.
                // 'PATH' can only be updated in a single shell block.
                // We can not update PATH in 'environment' block.
                // Pytest needs to export the junit report.
                withEnv(["HOME=${env.WORKSPACE}"]) {
                    sh """
                        source ${env.DEV_TOOL}
                        source ${env.LSST_STACK}/loadLSST.bash
                        cd phosim_utils/
                        setup -k -r . -t ${env.SIMS_VERSION}
                        cd ..
                        setup -k -r .
                        pytest --cov-report html --cov=${env.MODULE_NAME} --junitxml=${env.XML_REPORT} tests/
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
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: "Coverage Report"
              ])
        }

        cleanup {
            // clean up the workspace
            deleteDir()
        }
    }
}

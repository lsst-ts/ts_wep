@Library('JenkinsShared')_
DevelopPipeline(
    name: "ts_wep",
    module_name: "lsst.ts.wep",
    idl_names: [],
    build_all_idl: false,
    extra_packages: ["conda-forge/label/tarts_dev::tarts"],
    kickoff_jobs: [],
    slack_build_channel: "aos-builds",
    has_doc_site: true,
    require_git_lfs: true,
    require_scons: true
)

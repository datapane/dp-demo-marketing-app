import os

from datapane.package import Package, DuckDBConnector, RunTask, Email, Slack, Workflow, Task

# configure the package
package = Package(
    entrypoint="app.py",
    data_dir="data/",
    autodiscover=True,
    databases=[
        DuckDBConnector(file="data/data.db"),
    ],
    # package file configuration
    include=["a.py", "b/"],
    exclude=[],
    # scheduled tasks
    schedules=[
        Task(name="daily-report", cron="00 9 * * MON-FRI"),
    ],
    # notification configuration, e.g. Slack, Email. Teams
    notifications=[
        Email(smtp_config=os.environ["EMAIL_SMTP"]),
        Slack(slack_api_key=os.environ["SLACK_API_KEY"]),
    ],
    env_vars={},
    # ELT workflows
    workflows=[Workflow(name="Update DB", cron="00 1 * * *", source="shopify-tap", on_load="update-db")],
)


################################################################################
# Package lifecycle hooks
def first_run():
    """Code to run upon installing the package"""
    ...


def on_migration():
    """Run on updating the package"""
    ...

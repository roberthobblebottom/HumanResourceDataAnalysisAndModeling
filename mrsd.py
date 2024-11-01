import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def __(mo):
    mo.md(r"""#Import""")
    return


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import seaborn as sns
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_score,
        recall_score,
    )
    return (
        GradientBoostingClassifier,
        OrdinalEncoder,
        RandomForestClassifier,
        accuracy_score,
        average_precision_score,
        classification_report,
        mo,
        pl,
        precision_score,
        px,
        recall_score,
        roc_auc_score,
        sns,
    )


@app.cell
def __():
    from sklearn.model_selection import (
        StratifiedKFold,
        GridSearchCV,
        train_test_split,
    )
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    from imblearn.over_sampling import SMOTENC
    return (
        ColumnTransformer,
        GridSearchCV,
        Pipeline,
        SMOTENC,
        SelectKBest,
        StratifiedKFold,
        train_test_split,
    )


@app.cell
def __(mo):
    mo.md(r"""# Visualisation""")
    return


@app.cell
def __(mo):
    mo.md("""## There is alot of insights to go about here but I would like to know  employees satisfaction and engagement patterns""")
    return


@app.cell
def __(mo):
    mo.md(r"""## Checking and cleaning""")
    return


@app.cell
def __(pl):
    df_uncleaned = pl.read_csv("HRDataset_v14.csv")
    return (df_uncleaned,)


@app.cell
def __(df_uncleaned):
    df_uncleaned.columns
    return


@app.cell
def __(df_uncleaned):
    df_uncleaned["Department"].unique()
    return


@app.cell
def __(mo):
    mo.md(r"""### nans detected:""")
    return


@app.cell(hide_code=True)
def __(df_uncleaned, pl):
    (
        df_uncleaned.null_count()
        .transpose(
            include_header=True,
        )
        .filter(pl.col("column_0") > 0)
        .rename({"column": "feature", "column_0": "nan counts"})
    )
    return


@app.cell
def __(df_uncleaned):
    df_uncleaned["MaritalStatusID"]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        TODO: To clean:  
         - department string values spacing  
         - standardising naming convention in feature names. Verbosity in column names has increased for clarity
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(pl):
    df = pl.read_csv("HRDataset_v14.csv")
    df.columns = [
        "EmployeeName",
        "EmployeeID",
        "MarriedID",
        "MaritalStatusID",
        "GenderID",
        "EmpStatusID",
        "DepartmentID",
        "PerformanceScoreNumerical",
        "FromDiversityJobFairID",
        "Salary",
        "Termd",
        "PositionID",
        "Position",
        "State",
        "Zip",
        "DateOfBirth",
        "Sex",
        "MaritalDesc",
        "CitizenDesc",
        "HispanicLatino",
        "RaceDescent",
        "DateOfHire",
        "DateOfTermination",
        "TerminationReason",
        "EmploymentStatus",
        "Department",
        "ManagerName",
        "ManagerID",
        "RecruitmentSource",
        "PerformanceScore",
        "EngagementSurvey",
        "EmployeeSatisfaction",
        "SpecialProjectsCount",
        "LastPerformanceReviewDate",
        "DaysLateLast30Days",
        "Absences",
    ]
    df = df.with_columns(
        pl.col(
            "DateOfHire",
        )
        .str.to_date("%m/%d/%Y")
        .cast(pl.UInt64),
        pl.col("DateOfBirth").str.to_date("%m/%d/%y").cast(pl.UInt64),
        pl.col("DateOfTermination").str.to_date("%m/%d/%Y").cast(pl.UInt64),
        pl.col("LastPerformanceReviewDate")
        .str.to_date("%m/%d/%Y")
        .cast(pl.UInt64),
        pl.col("Department").str.strip_chars(),
        pl.col("ManagerID").fill_null(-1),
    ).with_columns(
        pl.col("DateOfTermination").fill_null(0),
    )
    return (df,)


@app.cell
def __(df, pl):
    (
        df.null_count()
        .transpose(
            include_header=True,
        )
        .filter(pl.col("column_0") > 0)
        .rename({"column": "feature", "column_0": "nan counts"})
    )
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df):
    df.sample(5)
    return


@app.cell
def __(df):
    df["PerformanceScore", "PerformanceScoreNumerical"].sample(5)
    return


@app.cell
def __(df):
    df["Department", "DepartmentID"].sample(5)
    return


@app.cell
def __(mo):
    mo.md(r"""### There are correspondence between ID suffixed features that is of int type and the feature that is of string type""")
    return


@app.cell
def __(mo):
    mo.md(r"""##  counts of positions which satisfaciton less than 3 and engagement less than 3""")
    return


@app.cell(hide_code=True)
def __(df, pl, px):
    _d = (
        df.filter(
            (pl.col("EmployeeSatisfaction") < 3) | (pl.col("EngagementSurvey") < 3)
        )
        .select(["Position", "EmploymentStatus", "TerminationReason"])
        .group_by("Position", "EmploymentStatus", "TerminationReason")
        .len()
        .sort("len")
    )
    position_order = _d["Position"].unique().sort()
    # print(position_order)
    px.bar(
        _d.to_pandas(),
        x="Position",
        y="len",
        color="EmploymentStatus",
        pattern_shape="TerminationReason",
        title=" counts of position with satisfaction less than 3 and engagement less than 3",
        category_orders={
            "EmploymentStatus": ["Voluntary Terminated", "Terminated", "Active"],
            "Position": position_order,
        },
    )
    return (position_order,)


@app.cell
def __(mo):
    mo.md(r"""##  counts of Manager whose employees under them have satisfaciton less than 3 and engagement less than 3""")
    return


@app.cell(hide_code=True)
def __(df, pl, px):
    _d = (
        df.filter(
            (pl.col("EmployeeSatisfaction") < 3) | (pl.col("EngagementSurvey") < 3)
        )
        .select(
            ["ManagerName", "Department", "EmploymentStatus", "TerminationReason"]
        )
        .group_by(
            "ManagerName", "Department", "EmploymentStatus", "TerminationReason"
        )
        .len()
        .with_columns(
            DepartmentAndManagerName=pl.concat_str(
                "Department", pl.lit(" "), "ManagerName"
            )
        )
        .sort("Department", "len")
    )
    departmentAndManagerName_order = _d["DepartmentAndManagerName"].unique().sort()
    # _managers_of_interest = _d["ManagerName"].unique()
    # _departments = df.filter(pl.col("EmployeeName").is_in(["David Stanley"])).select("Department")
    # print(_departments)
    px.bar(
        _d.to_pandas(),
        x="DepartmentAndManagerName",
        y="len",
        color="EmploymentStatus",
        pattern_shape="TerminationReason",
        title=" counts of Manager whose employees under them have satisfaction less than 3 and engagement less than 3",
        category_orders={
            "EmploymentStatus": ["Voluntary Terminated", "Terminated", "Active"],
            "DepartmentAndManagerName": departmentAndManagerName_order,
        },
    ).update_yaxes(tickvals=[x for x in range(4)]).update_xaxes(tickangle=45)
    return (departmentAndManagerName_order,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Scatter Plot Matrix and Correlations heatmap

        Selected scatter plot matrix
        """
    )
    return


@app.cell(hide_code=True)
def __(df):
    _cols = [
        "EmployeeID",
        "EmpStatusID",
        "DepartmentID",
        "PerformanceScoreNumerical",
        "FromDiversityJobFairID",
        "Salary",
        "Termd",
        "PositionID",
        "ManagerID",
        "EngagementSurvey",
        "EmployeeSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30Days",
        "Absences",
    ]

    _cols2 = [
        "EmpID",
        "EmpStatusID",
        "DepartmentID",
        "PerfScoreID",
        "DEIJobFair",
        "Salary",
        "Termd",
        "PositionID",
        "ManagerID",
        "Engagement",
        "Satisf",
        "SProjsCount",
        "DaysLate",
        "Absent",
    ]
    # _col_numbered = [str(x) for x in range(len(_cols))]
    _d = df.select(_cols)
    _d.columns = _cols2
    legend = "\n ".join([" = ".join(x) for x in zip(_cols, _cols2)])
    d1 = _d
    return d1, legend


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Legends below are of feature name to the number that is shown on the scatterplot matrix below. ScatterPlot Matrix is fast but I can't rotate the full feature name for some reason.""")
    return


@app.cell
def __(legend):
    legend
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""attempted to do scatter plot matrix to find other correlations""")
    return


@app.cell(hide_code=True)
def __(d1, px):
    px.scatter_matrix(
        d1.to_pandas(),
        width=1000,
        height=1000,
    )
    return


@app.cell
def __(mo):
    mo.md(r"""Removed ManagerID because for some reason it doesnt show up in the heatmaps""")
    return


@app.cell(hide_code=True)
def __(df, pl, px):
    _d = (
        df.select(pl.exclude(pl.String))
        .select(pl.exclude("ManagerID"))
        .corr()
        .select(pl.all().round(2))
    )

    d2 = _d
    px.imshow(_d, text_auto=True, aspect="auto", title="pearson").update_yaxes(
        tickmode="array",
        tickvals=[x for x in range(_d.shape[0])],
        ticktext=_d.columns,
    )
    return (d2,)


@app.cell(hide_code=True)
def __(d2, px):
    _d = d2.to_pandas().corr(method="spearman").round(2)
    px.imshow(
        _d,
        text_auto=True,
        aspect="auto",
        title="spearman",
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        correlations of interests :  


        performance score id | dayslatelast30  
        departmentid | special projects count  
        engagementStudy | PerformanceScoreNumerical  
        EmployeeSatisfaction | PerformanceScoreNumerical  
        Salary | special projects count    
        engagementStudy | dayslatelast30

        I used Spearman too just in case of the linear correlations comparision of person is misleading. Although to be fair the scatter plot doesn’t show anything very obvious linearity
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ##Performance vs Department
        getting departments with performance score less than 3
        """
    )
    return


@app.cell(hide_code=True)
def __(df, pl, px):
    _d = (
        df.with_columns(
            performance=pl.when(
                pl.col("PerformanceScoreNumerical").is_between(1, 2)
            )
            .then(pl.lit("score of 1 or 2"))
            .otherwise(pl.lit("score of 3 or 4"))
        )
        .group_by("Department", "performance")
        .len()
        # .with_columns(
        #     group=pl.concat_str(
        #         pl.col("Department"),
        #         pl.lit(" performance "),
        #         pl.col("PerformanceScoreNumerical"),
        #     )
        # )
        .sort("Department", "performance")
    )

    total_d = (
        _d.group_by("Department")
        .sum()
        .select("Department", "len")
        .rename({"len": "total"})
    )
    _d = _d.join(total_d, on="Department").with_columns(
        (pl.col("len") * 100 / pl.col("total")).round(2).alias("proportion %"),
        pl.col("performance").cast(pl.String),
    )
    # print(_d)
    px.bar(
        _d.to_pandas(),
        x="Department",
        y="proportion %",
        color="performance",
    )
    return (total_d,)


@app.cell(disabled=True, hide_code=True)
def __(mo):
    mo.md(r"""##Employee satisfaction vs Performance""")
    return


@app.cell(disabled=True, hide_code=True)
def __(df, px):
    px.box(
        df.sort(by=["EmployeeSatisfaction"]).to_pandas(),
        y="EmployeeSatisfaction",
        x="PerformanceScoreNumerical",
        # box=True,
        points="all",
        labels=dict(PerformanceScoreNumerical="Performance Score out of 4"),
    ).update_xaxes(
        tickvals=[1, 2, 3, 4],
    )
    return


@app.cell(disabled=True, hide_code=True)
def __(mo):
    mo.md(r"""# satisfaction and department""")
    return


@app.cell(disabled=True, hide_code=True)
def __(df, px):
    px.box(
        df.sort(by=["EmployeeSatisfaction"]).to_pandas(),
        y="EmployeeSatisfaction",
        x="Department",
        # box=True,
        points="all",
    )
    # .update_yaxes(tickvals=[1, 2, 3, 4])
    return


@app.cell(disabled=True, hide_code=True)
def __(mo):
    mo.md(r"""## Satisfaction and Salary""")
    return


@app.cell(disabled=True, hide_code=True)
def __(df, px):
    px.box(
        df.sort(by=["EmployeeSatisfaction"]).to_pandas(),
        x="EmployeeSatisfaction",
        y="Salary",
        # box=True,
        points="all",
    )
    # .update_yaxes(tickvals=[1, 2, 3, 4])
    return


@app.cell
def __(mo):
    mo.md(r"""## satisfaction vs engagement vs peformance""")
    return


@app.cell
def __(df, px):
    px.scatter(
        df.to_pandas(),
        x="EngagementSurvey",
        y="EmployeeSatisfaction",
        size="PerformanceScoreNumerical",
        color="PerformanceScoreNumerical",
        title="the size is represent performance differences",
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## satisfaction vs engagement vs Salary""")
    return


@app.cell
def __(df, px):
    px.scatter(
        df.to_pandas(),
        x="EngagementSurvey",
        y="EmployeeSatisfaction",
        color="Salary",
        size="Salary",
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## satisfaction vs engagement vs days late in the last 30 days""")
    return


@app.cell
def __(df, px):
    px.scatter(
        df.to_pandas(),
        x="EngagementSurvey",
        y="EmployeeSatisfaction",
        color="DaysLateLast30Days",
    )
    return


@app.cell
def __(mo):
    mo.md(r"""##Engagement vs  days late last 30 days""")
    return


@app.cell
def __(df, px):
    px.box(
        df.to_pandas(),
        y="EngagementSurvey",
        x="DaysLateLast30Days",
        points="all",
    )
    return


@app.cell
def __(df, px):
    px.box(
        df.to_pandas(),
        y="PerformanceScoreNumerical",
        x="DaysLateLast30Days",
        points="all",
        title="days late vs performance",
    )
    return


@app.cell(disabled=True, hide_code=True)
def __(mo):
    mo.md(r"""##  engagement vs salary""")
    return


@app.cell(disabled=True, hide_code=True)
def __(df, px):
    px.scatter(
        df.to_pandas(),
        x="EngagementSurvey",
        y="Salary",
        trendline="lowess",
        trendline_options=dict(frac=0.31),
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Findings
        Production, software engineering, sales department has the highest proportion of poor performers under the score of 3. 

        Production, software engineering, sales departments are the only departments who have satisfaction and engagement less than the score of 3 

        production has some volunteery resignations because of unhappiness

        Production and Sales and Software Engineering has the biggest proportion of poorer performers compared to others

        those who are not late at all will be at least average level of engagement

        Salary doesn’t seem to correlate to employee satifaction and engagment as much, there are many with low salaries that has high satisfaction and engagement

        higher the engagement, the higher the satisfaction
        more employee satifaction and engagment seems to produces better preformances

        those that are not late at all for the past 30 days have at least higher engagement.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Some reforms should be done for production department, they have a lot of disengaged, unsatisfied and likely unhappy people in there that causes resignations and poorer performances.

         Is the firing reasons of performance and absense with no good reasons based on their disengagement, unsatisfaction and unhappiness? More eda is needed on this

        Sales and software engineering has similar issues too but with less resignations due to unhappiness unless 

        Boosting salary may not help in the satisfaction and engement metrics.

        Days lates have that corelations to satifaction and engagment metric. Are they really hate their jobs that they rather be late to work?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""#hypothesis testing""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Modeling and evaluation for engagement and satisfaction 

        creation of label that averages out for engagment and satisfaction  

        As seen before some of the features are already encoded into "IDs"
        """
    )
    return


@app.cell
def __(SMOTENC, df, pl):
    _feature_names = [
        # "EmployeeName",
        # "EmployeeID",
        "MarriedID",
        "MaritalStatusID",
        "GenderID",
        "EmpStatusID",
        "DepartmentID",
        "PerformanceScoreNumerical",
        "FromDiversityJobFairID",
        "Salary",
        "PositionID",
        "Zip",
        "DateOfBirth",
        "CitizenDesc",
        "DateOfHire",
        "DateOfTermination",
        # "TerminationReason",
        "EmploymentStatus",
        "ManagerID",
        "EngagementSurvey",
        "EmployeeSatisfaction",
        "SpecialProjectsCount",
        "LastPerformanceReviewDate",
        "DaysLateLast30Days",
        "Absences",
    ]  # 22 features
    print(len(_feature_names))
    features = df.select(_feature_names)

    labels = features["EmployeeSatisfaction"]
    features = features.select(pl.exclude("EmployeeSatisfaction"))
    sm = SMOTENC(
        categorical_features=features.select(pl.col(pl.String)).columns,
        k_neighbors=1,
    )
    (features, labels) = sm.fit_resample(features.to_pandas(), labels.to_pandas())
    features = pl.from_pandas(features)
    return features, labels, sm


@app.cell
def __(features):
    features
    return


@app.cell
def __(mo):
    mo.md(r"""### balanced out with SMOTE""")
    return


@app.cell
def __(labels, pl):
    pl.from_pandas(labels).to_frame().group_by("EmployeeSatisfaction").len()
    return


@app.cell
def __(StratifiedKFold):
    skfold = StratifiedKFold(n_splits=5)
    return (skfold,)


@app.cell
def __(mo):
    mo.md(r"""## Random Forest Classifier without grid search""")
    return


@app.cell
def __(
    ColumnTransformer,
    OrdinalEncoder,
    Pipeline,
    RandomForestClassifier,
    SelectKBest,
    accuracy_score,
    average_precision_score,
    features,
    labels,
    pl,
    precision_score,
    recall_score,
    roc_auc_score,
    skfold,
):
    metrics = pl.DataFrame()
    ranks = pl.DataFrame()
    pipeline = Pipeline(
        [
            (
                "ordinal_encoder_column_transformer",
                ColumnTransformer(
                    [
                        (
                            "ordinal_encoder",
                            OrdinalEncoder(),
                            features.select(pl.col(pl.String)).columns,
                        ),
                        (
                            "passthrough",
                            "passthrough",
                            features.select(pl.exclude(pl.String)).columns,
                        ),
                    ]
                ),
            ),
            ("select_k_best", SelectKBest(k=11)),
            ("random_forest_classifier", RandomForestClassifier(random_state=0)),
        ]
    )
    for i, (train_index, test_index) in enumerate(skfold.split(features, labels)):
        test_labels = labels[test_index]

        pipeline.fit(features[train_index].to_pandas(), labels[train_index])
        # print(features[test_index].to_pandas())
        prediction = pipeline.predict(features[test_index].to_pandas())
        prediction_probabilities = pipeline.predict_proba(
            features[test_index].to_pandas()
        )

        t = pipeline.named_steps["select_k_best"].get_support()
        names = features[t].columns
        importance = pipeline.named_steps[
            "random_forest_classifier"
        ].feature_importances_

        _d = pl.DataFrame(
            [names, importance], schema=["features", "importance"]
        ).sort("importance", descending=True)
        ranks = ranks.hstack(
            _d["features"]
            .to_frame(name=str(i))
            .with_columns(rank=range(_d.shape[0]))
            .rename(dict(rank="rank" + str(i)))
            .sort(str(i))
        )

        metrics = metrics.vstack(
            pl.DataFrame(
                {
                    "accuracy": accuracy_score(test_labels, prediction),
                    "recall": recall_score(
                        test_labels, prediction, average="macro"
                    ),
                    "precision": precision_score(
                        test_labels, prediction, average="macro"
                    ),
                    "auc": roc_auc_score(
                        test_labels,
                        prediction_probabilities,
                        average="macro",
                        multi_class="ovr",
                    ),
                    "pr_auc": average_precision_score(
                        test_labels, prediction_probabilities, average="macro"
                    ),
                }
            ).with_columns(pl.all().round(2))
        )
    return (
        i,
        importance,
        metrics,
        names,
        pipeline,
        prediction,
        prediction_probabilities,
        ranks,
        t,
        test_index,
        test_labels,
        train_index,
    )


@app.cell
def __(metrics):
    metrics.mean()
    return


@app.cell
def __(mo):
    mo.md(r"""## rank of feature importance on average""")
    return


@app.cell
def __(pl, ranks):
    ranks.select(["0"]).rename({"0": "feature"}).hstack(
        (ranks.select(pl.col(pl.Int64)).sum_horizontal() / 11)
        .to_frame(name="rank")
        .select(pl.all().round(2))
    ).sort("rank", descending=True)["feature"].to_list()
    return


@app.cell
def __(mo):
    mo.md(r"""## GridSearchCrossValidation attempt""")
    return


@app.cell
def __(GridSearchCV, features, labels, pipeline, train_test_split):
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, random_state=0, test_size=0.1
    )
    gs = GridSearchCV(
        pipeline,
        param_grid={
            "select_k_best__k": [11, 16],
            "random_forest_classifier__n_estimators": [50, 100],
            # "random_forest_classifier__max_features": ["sqrt", "log2", None],
            "random_forest_classifier__max_depth": [3, 6, 9],
            "random_forest_classifier__max_leaf_nodes": [3, 6, 9],
        },
    )
    gs.fit(features_train, labels_train)
    print(gs.best_estimator_)
    # print(features[test_index].to_pandas())
    # prediction = gs.predict(features[test_index].to_pandas())
    # prediction_probabilities = gs.predict_proba(features[test_index].to_pandas())
    return features_test, features_train, gs, labels_test, labels_train


@app.cell
def __(
    accuracy_score,
    average_precision_score,
    features,
    features_test,
    features_train,
    labels_test,
    labels_train,
    pipeline,
    pl,
    precision_score,
    recall_score,
    roc_auc_score,
    t,
):
    _metrics = pl.DataFrame()
    _ranks = pl.DataFrame()
    _pipeline = pipeline.set_params(
        **{
            "select_k_best__k": 11,
            "random_forest_classifier__n_estimators": 50,
            "random_forest_classifier__max_depth": 6,
            "random_forest_classifier__max_leaf_nodes": 6,
        }
    )
    _pipeline.fit(features_train, labels_train)
    # print(features[test_index].to_pandas())
    _prediction = _pipeline.predict(features_test)
    _prediction_probabilities = _pipeline.predict_proba(features_test)

    _t = _pipeline.named_steps["select_k_best"].get_support()
    _names = features[t].columns
    _importances = _pipeline.named_steps[
        "random_forest_classifier"
    ].feature_importances_

    _d = pl.DataFrame(
        [_names, _importances], schema=["features", "importance"]
    ).sort("importance", descending=True)
    ranks_after_gs = (
        _d["features"]
        .to_frame()
        .with_columns(rank=range(_d.shape[0]))
        .sort("rank", descending=True)
    )

    # print(label_test)
    # print()
    metrics_after_gs = pl.DataFrame(
        {
            "accuracy": accuracy_score(labels_test, _prediction),
            "recall": recall_score(labels_test, _prediction, average="macro"),
            "precision": precision_score(
                labels_test, _prediction, average="macro"
            ),
            "auc": roc_auc_score(
                labels_test,
                _prediction_probabilities,
                average="macro",
                multi_class="ovr",
            ),
            "pr_auc": average_precision_score(
                labels_test, _prediction_probabilities, average="macro"
            ),
        }
    ).with_columns(pl.all().round(2))
    return metrics_after_gs, ranks_after_gs


@app.cell
def __(ranks_after_gs):
    ranks_after_gs.to_series().to_list()
    return


@app.cell
def __(metrics_after_gs):
    metrics_after_gs
    return


@app.cell
def __(mo):
    mo.md(r"""There are a bit of improvement overall after grid search""")
    return


if __name__ == "__main__":
    app.run()

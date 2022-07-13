import pickle
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from shapely.geometry import LineString
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import participant


def tune_svm(df):
    """Tunes an SVM for the data given using cross validation
    :arg
        df (dataFrame): genuine participant and imposter participant features combined in dataFrame
    :return
        the best performing/tuned model
    """
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # parameters to train the SVM
    parameters = {"gamma": [1, 0.1, 0.01, 0.001, 0.0001],
                  "C": [0.1, 1, 10, 100, 1000],
                  "kernel": ["rbf", "linear", "poly"]}

    # grid search that tries all parameter combinations
    grid_search = GridSearchCV(SVC(probability=True), parameters, refit=True, verbose=1)

    # fitting the model for grid search
    grid_search.fit(X_train, y_train)

    # get the svm with tuned parameters
    best_svm = grid_search.best_estimator_
    return best_svm


def tune_dt(df):
    """Tunes an Decision Eree for the data given using cross validation
    :arg
        df (dataFrame): genuine participant and imposter participant features combined in dataFrame
    :return
        the best performing/tuned model
    """
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # parameters to train the decision tree
    parameters = {"splitter": ["best", "random"],
                  "max_depth": [None, 1, 4, 8, 12],
                  "min_samples_leaf": [1, 2, 4, 6, 8, 10],
                  "min_weight_fraction_leaf": [0, 0.2, 0.4],
                  "max_features": ["auto", "log2", "sqrt", None],
                  "max_leaf_nodes": [None, 20, 40, 60, 80]}

    # grid search that tries all parameter combinations
    grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, refit=True, verbose=1)

    # fitting the model for grid search
    grid_search.fit(X_train, y_train)

    # get the decision tree with tuned parameters
    best_dt = grid_search.best_estimator_
    return best_dt


def get_models():
    """Tunes each model on every combination of genuine and imposters for each stage.
    Also pickles the participant object once all models are tuned and saved"""

    imposter_dict = {}  # key is the genuine user and value is list of imposters
    # get list of possible imposter participant IDs
    for participant in participants:
        possible_imposters = participants.copy()
        possible_imposters.remove(participant)
        imposter_dict[participant] = possible_imposters
        participant.import_file_data()

    for stage in STAGES:
        for participant in participants:
            participant.generate_all_features(stage)

        for participant, imposters in imposter_dict.items():
            for imposter in imposters:
                df1 = pd.DataFrame.from_dict(participant.features)
                df2 = pd.DataFrame.from_dict(imposter.features)
                df = df1.append(df2, ignore_index=True)  # combine genuine and imposter features
                # tune the models
                svm = tune_svm(df)
                dt = tune_dt(df)
                participant.save_model(svm, dt, imposter.id)
            file_name = participant.id + "_" + str(stage) + "_object.pkl"
            # save object for each participant for later use
            with open(file_name, "wb") as file:
                pickle.dump(participant, file)


def import_objects(participants):
    """Imports the participant objects with tuned models from the pickle files
    :arg
        participants (list): all participant objects to get their ID
    :return
        all ten participant objects
    """
    all_objects = defaultdict(list)  # key is the id and values are different stage objects
    for stage in STAGES:
        for participant in participants:
            id = participant.id
            path = id + "\\" + id + "_" + str(stage) + "_object.pkl"
            with open(path, "rb") as file:
                print("Loading participant", id, "stage:", stage)
                all_objects[id].append(pickle.load(file))
        print()
    print("Import complete")
    return all_objects


def svm_eer(participant, stage, remove):
    """Calcualte the FAR, FRR and EER with different thresholds for the genuine participant against each imposter and scatter plot graphs
    :arg
        participant (Participant object): genuine participant object
        stage (int): the stage number
        remove (boolean): true if the 10 initial drawings should be removed and false if no data to be removed
    :return
        list of EERs for the participant
    """
    svms = participant.svms  # all tuned SVM models
    eers = []
    for imposter_id, svm in svms.items():
        imposter = get_imposter(imposter_id, stage, remove)

        # remove data
        if remove:
            participant.import_file_data()
            participant.remove_data(stage, 10)
            participant.generate_all_features(stage)

        df1 = pd.DataFrame.from_dict(participant.features)
        df2 = pd.DataFrame.from_dict(imposter.features)
        df = df1.append(df2, ignore_index=True)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        # 25% split into test/training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc_X = StandardScaler()  # scale the features
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        svm.probability = True
        svm.fit(X_train, y_train)  # train model
        probs = svm.predict_proba(X_test)  # get label probabilites
        eer, frrs, fars = get_values(participant, imposter, y_test, probs)  # get EER, FRR, FAR

        # plot data
        plt.xlabel("Threshold (%)")
        plt.ylabel("Error Rate (%)")
        plt.title("Stage " + str(stage) + ":  Genuine-" + participant.id + "  Imposter-" + imposter.id)
        plt.plot(range(0, 101, 5), frrs, label="FRR")
        plt.plot(range(0, 101, 5), fars, label="FAR")
        plt.xticks(range(0, 101, 10))
        plt.yticks(range(0, 101, 10))
        plt.text(107, eer, "EER: " + str(eer) + "%")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        file_name = str(stage) + "_" + participant.id + "_" + imposter.id + "_authentication.png"
        plt.savefig(participant.id + "\\" + file_name, bbox_inches="tight")
        plt.clf()

        eers.append(eer)

    return eers


def dt_eer(participant, stage):
    """Calcualte the FAR, FRR and EER with different thresholds for the genuine participant against each imposter
    :arg
        participant (Participant object): genuine participant object
        stage (int): the stage number
    :return
        list of EERs for the participant
    """
    dts = participant.dts  # all tuned decision trees models
    eers = []
    for imposter_id, dt in dts.items():
        imposter = get_imposter(imposter_id, stage, False)
        df1 = pd.DataFrame.from_dict(participant.features)
        df2 = pd.DataFrame.from_dict(imposter.features)
        df = df1.append(df2, ignore_index=True)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        # 25% split into test/training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc_X = StandardScaler()  # scale the features
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        dt.probability = True
        dt.fit(X_train, y_train)  # train model
        probs = dt.predict_proba(X_test)  # get label probabilities
        eer, frrs, fars = get_values(participant, imposter, y_test, probs)  # get EER, FRR, FAR
        eers.append(eer)

    return eers


def nb_eer(participant, stage):
    """Calcualte the FAR, FRR and EER with different thresholds for the genuine participant against each imposter
    :arg
        participant (Participant object): genuine participant object
        stage (int): the stage number
    :return
        list of EERs for the participant
    """
    svms = participant.svms
    eers = []
    for imposter_id in svms.keys():
        imposter = get_imposter(imposter_id, stage, False)
        df1 = pd.DataFrame.from_dict(participant.features)
        df2 = pd.DataFrame.from_dict(imposter.features)
        df = df1.append(df2, ignore_index=True)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        nb = GaussianNB()  # initialise new model
        nb.fit(X_train, y_train)  # train model
        probs = nb.predict_proba(X_test)
        eer, frrs, fars = get_values(participant, imposter, y_test, probs)  # get EER, FRR, FAR
        eers.append(eer)

    return eers


def get_values(participant, imposter, y_test, probs):
    """Calcualte the FAR, FRR and EER with different thresholds
    :arg
        participant (Participant object): genuine participant
        imposter (Participant object): imposter participant
        stage (int): the stage number
    :return
        the eer, ffrs and fars
    """
    thresholds = np.arange(0.0, 1.01, 0.05)
    predict_labels = []
    fars = []
    frrs = []
    # get all predictions by model based on the threshold
    for threshold in thresholds:
        predict_labels.append(get_labels(threshold, [participant.id, imposter.id], probs))

    # calculate the FAR, FRR and EER
    for labels in predict_labels:
        tn, fp, fn, tp = confusion_matrix(y_test, labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        fars.append(far * 100)
        frrs.append(frr * 100)

    ffr_line = LineString(np.column_stack((range(0, 101, 5), frrs)))
    far_line = LineString(np.column_stack((range(0, 101, 5), fars)))
    intersection = ffr_line.intersection(far_line)  # find where lines intersect (eer)
    if intersection.geom_type == "MultiLineString":
        eer = round(list(intersection)[0].coords[0][1], 2)
    elif intersection.geom_type == "Point":
        eer = round(list(intersection.coords)[0][1], 2)
    elif intersection.geom_type == "LineString":
        if intersection.is_empty:
            eer = None
        else:
            eer = round(list(intersection.coords)[0][1], 2)

    return eer, frrs, fars


def get_labels(threshold, labels, probs):
    """Get the label predicitons based on the probabilities and threshold
    :arg
        threshold (float): threshold percentage
        labels (list): the two possible labels (genuine and imposter IDs)
        probs (list): all probabilities from model
    :return
        list of prediciton labels
    """
    predictions = []
    for prob in probs:
        if prob[0] > threshold:
            predictions.append(labels[0])
        else:
            predictions.append(labels[1])

    return predictions


def get_imposter(id, stage, remove):
    """Get the imposter participant object from participant ID and remove data if specified
    :arg
        id (string): imposter participant ID
        stage (int): the stage number
        remove (boolean): true if the 10 initial drawings should be removed and false if no data to be removed
    :return
        imposter participant object
    """
    imposter = participant.Participant(id)
    imposter.import_file_data()
    if remove:
        imposter.remove_data(stage, 10)
    imposter.generate_all_features(stage)
    return imposter


def eers_all_models(eers_svm, eers_dt, eers_nb):
    """Generates the bar graphs for the EERs across models
    :arg
        eers_svm (dict): key as stage and value as list of all EERs from the SVM models
        eers_dt (dict): key as stage and value as list of all EERs from the Decision Tree models
        eers_nb (dict): key as stage and value as list of all EERs from the Naive Bayes models
    """

    # generate bar graph for mean EER across all stages for each model
    eers_all_stages_svm = []
    eers_all_stages_dt = []
    eers_all_stages_nb = []

    for i in range(len(eers_svm)):
        for value in eers_svm[i].values():
            eers_all_stages_svm += value
        for value in eers_dt[i].values():
            eers_all_stages_dt += value
        for value in eers_nb[i].values():
            eers_all_stages_nb += value

    plt.xlabel("Classification models", labelpad=10)
    plt.ylabel("Equal Error Rate (EER) (%)")
    plt.title("Mean EERs across all participants and stages of the\n "
              "SVM, Decision Tree and Naive Bayes classification models")
    labels = ["SVM", "Decision Tree", "Naive Bayes"]

    # calculate means
    means = [statistics.mean(eers_all_stages_svm), statistics.mean(eers_all_stages_dt),
             statistics.mean(eers_all_stages_nb)]

    for i in range(len(means)):
        plt.bar(labels[i], means[i], label=labels[i], width=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.savefig("mean_eers_all_models.png", bbox_inches="tight")
    plt.clf()

    #  generate bar  for mean EER across for each stages for each model
    bars = []
    for i in range(len(eers_svm)):
        eers_current_stage_svm = []
        eers_current_stage_dt = []
        eers_current_stage_nb = []
        for value in eers_svm[i].values():
            eers_current_stage_svm += value
        for value in eers_dt[i].values():
            eers_current_stage_dt += value
        for value in eers_nb[i].values():
            eers_current_stage_nb += value

        bars.append([statistics.mean(eers_current_stage_svm), statistics.mean(eers_current_stage_dt),
                     statistics.mean(eers_current_stage_nb)])

    labels = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    models = ["SVM", "Decision Tree", "Naive Bayes"]
    plt.xlabel("Classification models", labelpad=10)
    plt.ylabel("Equal Error Rate (EER) (%)")
    plt.title("Mean EERs across all participants of the\n "
              "SVM, Decision Tree and Naive Bayes classification models")

    x_axis = np.arange(len(models))

    for i in range(len(bars)):
        plt.bar(x_axis + 0.20 * (i + 1), bars[i], width=0.2, label=labels[i])

    plt.xticks(x_axis + 0.5, models)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.savefig("mean_eers_all_models_stages.png", bbox_inches="tight")
    plt.clf()


def eer_matrix(all_eers):
    """Plot confusion matrix of eers using heatmap
    :arg
        all_eers (dict): key as stage and value as list of all EERs from the SVM models
    """
    labels = list(all_eers[0].keys())  # labels for plot

    # generate matrix for mean EER across all stages for each participant
    data = []
    for key in labels:
        current_data = []
        for i in range(len(all_eers[0][key])):
            to_mean = [all_eers[0][key][i], all_eers[1][key][i], all_eers[2][key][i], all_eers[3][key][i]]
            mean = statistics.mean(to_mean)
            current_data.append(mean)

        data.append(current_data)

    for i in range(len(data)):
        data[i].insert(i, -1)

    seaborn.set(color_codes=True)
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="Blues", cbar_kws={"label": "Equal Error Rate (%)"})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)

    ax.set(ylabel="Genuine", xlabel="Imposter")
    plt.title("EER matrix genuine and imposter\ncombinations meand across stages", pad=10)
    plt.savefig("eer_matrix.png", bbox_inches="tight")
    plt.clf()

    # generate matrix for EER across all stages 1 and 4 for each participant
    for stage in [1, 4]:
        data = []
        for key in labels:
            current_data = []
            for i in range(len(all_eers[stage - 1][key])):
                current_data.append(all_eers[stage - 1][key][i])

            data.append(current_data)

        for i in range(len(data)):
            data[i].insert(i, -1)

        seaborn.set(color_codes=True)
        seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="Blues", cbar_kws={"label": "Equal Error Rate (%)"})

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)

        ax.set(ylabel="Genuine", xlabel="Imposter")
        plt.title("EER matrix genuine and imposter\ncombinations for stage " + str(stage), pad=10)
        plt.savefig("eer_matrix_" + str(stage) + ".png", bbox_inches="tight")
        plt.clf()


def eer_across_stages(all_eers):
    """Plot bar graph of mean EER across all participants for each stage and with first 10 drawings removed
    :arg
        all_eers (dict): key as stage and value as list of all EERs from the SVM models
    """
    stage_1 = np.asarray(list(all_eers[0].values())).flatten()
    stage_2 = np.asarray(list(all_eers[1].values())).flatten()
    stage_3 = np.asarray(list(all_eers[2].values())).flatten()
    stage_4 = np.asarray(list(all_eers[3].values())).flatten()
    means = [statistics.mean(stage_1), statistics.mean(stage_2),
             statistics.mean(stage_3), statistics.mean(stage_4)]

    labels = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    plt.xlabel("Stage", labelpad=10)
    plt.ylabel("Equal Error Rate (EER) (%)")
    plt.title("Mean EERs across all participants for each stage")

    for i in range(len(means)):
        plt.bar(labels[i], means[i], label=labels[i], width=0.4)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.savefig("eer_across_stages", bbox_inches="tight")
    plt.clf()

    bars = [means]
    eers_stage_1_svm = {}
    eers_stage_2_svm = {}
    eers_stage_3_svm = {}
    eers_stage_4_svm = {}
    # re-calculate EER for models trained with data removed
    for i in range(len(stage_1_objects)):
        stage_1 = svm_eer(stage_1_objects[i], 1, True)
        stage_2 = svm_eer(stage_2_objects[i], 2, True)
        stage_3 = svm_eer(stage_3_objects[i], 3, True)
        stage_4 = svm_eer(stage_4_objects[i], 4, True)

        eers_stage_1_svm[stage_1_objects[i].id] = stage_1
        eers_stage_2_svm[stage_2_objects[i].id] = stage_2
        eers_stage_3_svm[stage_3_objects[i].id] = stage_3
        eers_stage_4_svm[stage_4_objects[i].id] = stage_4

    stage_1 = np.asarray(list(eers_stage_1_svm.values())).flatten()
    stage_2 = np.asarray(list(eers_stage_2_svm.values())).flatten()
    stage_3 = np.asarray(list(eers_stage_3_svm.values())).flatten()
    stage_4 = np.asarray(list(eers_stage_4_svm.values())).flatten()
    bars.append([statistics.mean(stage_1), statistics.mean(stage_2),
                 statistics.mean(stage_3), statistics.mean(stage_4)])

    x_labels = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    labels = ["1-50 drawings", "11-50 drawings"]
    plt.ylabel("Equal Error Rate (EER) (%)")
    plt.xlabel("Stage", labelpad=10)
    plt.title("Mean EERs across all participants with all drawing attempts\nand with the first ten attempts removed")

    x_axis = np.arange(len(x_labels))

    for i in range(len(bars)):
        plt.bar(x_axis + 0.20 * (i + 1), bars[i], width=0.2, label=labels[i])

    plt.xticks(x_axis + 0.29, x_labels)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.savefig("eers_different_drawing_number.png", bbox_inches="tight")
    plt.clf()


def mean_time(participants):
    """Calculates and plots the mean time line graphs
    :arg
        participants (list): all participant objects
    """
    stage_labels = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    line_labels = ["Stage 1 trend line", "Stage 2 trend line", "Stage 3 trend line", "Stage 4 trend line"]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    all_timings = defaultdict(list)
    # plot drawing time for each stage for each participant
    for participant in participants:
        participant.import_file_data()
        participant_timings = []
        for stage in STAGES:
            participant.generate_all_features(stage)
            timings = participant.all_timings
            counter = 1
            current_timings = []
            for timing in timings:
                current_timings.append([counter, timing])
                counter += 1

            participant_timings.append(np.asarray(current_timings))

        # plot times
        for i in range(len(STAGES)):
            x = participant_timings[i][:, 0]
            y = participant_timings[i][:, 1]
            all_timings[i].append(y)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, y, label=stage_labels[i], color=colours[i])  # plot time
            plt.plot(x, p(x), label=line_labels[i], color=colours[i], linestyle="--", alpha=0.7)  # plot trend line

        plt.xlabel("Drawing number")
        plt.ylabel("Time Taken to draw pattern (ms)")
        plt.xticks([1] + list(range(5, 51, 5)))
        plt.title("Participant " + participant.id + "'s time to complete each drawing for each stage")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        file_name = participant.id + "_time.png"
        plt.savefig(file_name, bbox_inches="tight")
        plt.clf()

    # plot mean drawing time for each stage across all participants
    for i in range(len(STAGES)):
        y = []
        values = list(zip(*all_timings[i]))
        for times in values:
            y.append(statistics.mean(times))

        x = range(1, len(y) + 1)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, y, label=stage_labels[i], color=colours[i])  # plot time
        plt.plot(x, p(x), label=line_labels[i], color=colours[i], linestyle="--", alpha=0.7)  # plot trend line

    plt.xlabel("Drawing number")
    plt.ylabel("Time Taken to draw pattern (ms)")
    plt.xticks([1] + list(range(5, 51, 5)))
    plt.title("Mean of all participant times for each stage")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.savefig("all_time.png", bbox_inches="tight")
    plt.show()
    plt.clf()


def best_features(participants):
    """Calculates k scores for all features for each stage
    :arg
        participants (list): all participant objects
    """
    for stage in STAGES:
        all_data = pd.DataFrame()
        # combine all features from all participants into one dataFrame
        for participant in participants:
            participant.import_file_data()
            participant.generate_all_features(stage)
            data = pd.DataFrame.from_dict(participant.features)
            all_data = all_data.append(data, ignore_index=True)

        all_data = all_data.fillna(0)
        X = all_data.iloc[:, 1:]
        y = all_data.iloc[:, 0]
        feature_names = pd.DataFrame(X.columns)
        sc_X = StandardScaler()  # scaler the features
        X = sc_X.fit_transform(X)

        selector = SelectKBest(score_func=f_classif, k="all")
        scores = round(pd.DataFrame(selector.fit(X, y).scores_), 2)  # get scores and round to 2 decimal points
        scores_df = pd.concat([feature_names, scores], axis=1)
        scores_df.columns = ["Feature", "Score"]  # naming the dataFrame columns
        print(scores_df.nlargest(5, "Score"))  # printing 10 best features
        print(scores_df.nsmallest(5, "Score"))  # printing 10 worst features


def compare_models():
    """Gets EERs for each model and calls appropriate methods to generate all graphs needed"""
    eers_stage_1_svm = {}
    eers_stage_2_svm = {}
    eers_stage_3_svm = {}
    eers_stage_4_svm = {}
    eers_stage_1_dt = {}
    eers_stage_2_dt = {}
    eers_stage_3_dt = {}
    eers_stage_4_dt = {}
    eers_stage_1_nb = {}
    eers_stage_2_nb = {}
    eers_stage_3_nb = {}
    eers_stage_4_nb = {}

    for i in range(len(stage_1_objects)):
        stage_1 = svm_eer(stage_1_objects[i], 1, False)
        stage_2 = svm_eer(stage_2_objects[i], 2, False)
        stage_3 = svm_eer(stage_3_objects[i], 3, False)
        stage_4 = svm_eer(stage_4_objects[i], 4, False)

        eers_stage_1_svm[stage_1_objects[i].id] = stage_1
        eers_stage_2_svm[stage_2_objects[i].id] = stage_2
        eers_stage_3_svm[stage_3_objects[i].id] = stage_3
        eers_stage_4_svm[stage_4_objects[i].id] = stage_4

        stage_1 = dt_eer(stage_1_objects[i], 1)
        stage_2 = dt_eer(stage_2_objects[i], 2)
        stage_3 = dt_eer(stage_3_objects[i], 3)
        stage_4 = dt_eer(stage_4_objects[i], 4)

        eers_stage_1_dt[stage_1_objects[i].id] = stage_1
        eers_stage_2_dt[stage_2_objects[i].id] = stage_2
        eers_stage_3_dt[stage_3_objects[i].id] = stage_3
        eers_stage_4_dt[stage_4_objects[i].id] = stage_4

        stage_1 = nb_eer(stage_1_objects[i], 1)
        stage_2 = nb_eer(stage_2_objects[i], 2)
        stage_3 = nb_eer(stage_3_objects[i], 3)
        stage_4 = nb_eer(stage_4_objects[i], 4)

        eers_stage_1_nb[stage_1_objects[i].id] = stage_1
        eers_stage_2_nb[stage_2_objects[i].id] = stage_2
        eers_stage_3_nb[stage_3_objects[i].id] = stage_3
        eers_stage_4_nb[stage_4_objects[i].id] = stage_4

    all_eers_svm = [eers_stage_1_svm, eers_stage_2_svm, eers_stage_3_svm, eers_stage_4_svm]
    all_eers_dt = [eers_stage_1_dt, eers_stage_2_dt, eers_stage_3_dt, eers_stage_4_dt]
    all_eers_nb = [eers_stage_1_nb, eers_stage_2_nb, eers_stage_3_nb, eers_stage_4_nb]
    eer_matrix(all_eers_svm)
    eer_across_stages(all_eers_svm)
    eer_across_stages(all_eers_svm)
    eers_all_models(all_eers_svm, all_eers_dt, all_eers_nb)


if __name__ == "__main__":
    STAGES = [1, 2, 3, 4]

    # create initial participant objects
    parti_A = participant.Participant("A")
    parti_B = participant.Participant("B")
    parti_C = participant.Participant("C")
    parti_D = participant.Participant("D")
    parti_E = participant.Participant("E")
    parti_F = participant.Participant("F")
    parti_G = participant.Participant("G")
    parti_H = participant.Participant("H")
    parti_I = participant.Participant("I")
    parti_J = participant.Participant("J")

    participants = [parti_A, parti_B, parti_C, parti_D, parti_E,
                    parti_F, parti_G, parti_H, parti_I, parti_J]

    participant_objects = import_objects(participants)
    all_objects = participant_objects.values()
    stage_1_objects, stage_2_objects, stage_3_objects, stage_4_objects = zip(*all_objects)
    compare_models()
    best_features(participants)
    for stage in STAGES:
        for participant in participants:
            participant.import_file_data()
            participant.plot_data(stage)
            participant.chunking(stage)

    mean_time(participants)

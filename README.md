# Biometric-and-cognitive-graphical-authentication
## Data collection
Java program using Swing to record x and y mouse coordinate data and timings as a user completes authentication patterns.
Data recorded in txt in the format:
x | y | time | indication of new node connected

NOTE: Create your own mp4 instructional video to instruct users on how to use the software and name it "video.mp4"

## Data analysis
Python program using scikit-learn and matplotlib to authenticate users and plotting analysis graphs. Analysis of the coordinate data performed using a trained SVM classifier.

## Research using this project
Abstract from research paper into using biometric and cognitive measures for graphical authentication conducted using this project:

**Pattern drawing lock screens are a prevalent method of user authentication used across
many mobile devices. Easy to recall, simple patterns are commonly selected by users
which consequently leads to poor security. A proposal to fix this issue is to incorporate
cognitive and biometric measures into the authentication system.
This report investigates using cognitive and biometric measures gathered from pattern
drawing to authenticate users. These measures were generated from participants acting as
both the genuine user and an imposter. Through the use of machine learning and nonmachine learning techniques, analysis was performed to identify whether participants can
be told apart based on how they draw a pattern. Our research found an overall mean
optimum False Rejection Rate (FRR) and False Acceptance Rate (FAR) of ~7%.**
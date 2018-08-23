# Gomoku-ML
Machine learned Convolutional Neural Network to learn how to play Gomoku

Since all my knowledge on Machine learning and AI is self learned over the course of this summer, the model still
seems to converge on some unusual behavior. Right now, after letting the model train (on an Nvidia GTX 1070, to give
and idea on the speed of training) for a few hours, the model continues to want to fill up the board, row by row first.

Right now, I have built the Convolutional layers based on a scholarly report
on CNN use in Gomoku: http://cs231n.stanford.edu/reports/2016/pdfs/109_Report.pdf

and the densely connected layers based on YouTube creator SentDex's videos on
AI learning the Cart Pole Game

Some improvements for the future:
  - Implement a tree search for the model
  - Test out different parameters for the Convolutional layers
  - Implement a better scoring function

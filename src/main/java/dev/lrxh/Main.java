package dev.lrxh;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static double weight1_1 = 0;
    public static double weight2_1 = 0;

    public static void main(String[] args) {

        // input1, input2, expectedOutput
        List<double[]> trainingData = Arrays.asList(
                new double[]{1, 1, 2},
                new double[]{1, 2, 3},
                new double[]{2, 2, 4},
                new double[]{2, 3, 5},
                new double[]{3, 1, 4},
                new double[]{3, 2, 5},
                new double[]{0, 0, 0},
                new double[]{0, 1, 1},
                new double[]{1, 0, 1},
                new double[]{2, 1, 3},
                new double[]{1, 3, 4},
                new double[]{3, 0, 3},
                new double[]{5, 5, 10},
                new double[]{6, 4, 10}
        );

        double[] weights = train(trainingData, 0.1, 100);
        System.out.println("Trained weights: weight1_1 = " + weight1_1 + ", weight2_1 = " + weight2_1);

        double input1 = 9;
        double input2 = 4;

        System.out.println("Output " + (neuralNetwork(input1, weights[0], input2, weights[1])));
    }

    private static double neuralNetwork(double input1, double weight1_1, double input2, double weight2_1) {
        return input1 * weight1_1 + input2 * weight2_1;
    }

    private static double[] train(List<double[]> data, double learningRate, double maxTries) {
        double w1 = weight1_1;
        double w2 = weight2_1;
        int epoch = 0;

        while (maxTries > epoch) {
            epoch++;
            double totalError = 0;

            for (double[] datum : data) {
                double input1 = datum[0];
                double input2 = datum[1];
                double expectedOutput = datum[2];

                //Calculate the error
                double output1 = input1 * w1 + input2 * w2;
                double error = expectedOutput - output1;
                totalError += error * error;

                //Gradient descent
                w1 += learningRate * error * input1;
                w2 += learningRate * error * input2;
            }

            System.out.println("Try #" + epoch + ", Total Error: " + totalError);

            if (totalError == 0) {
                weight1_1 = w1;
                weight2_1 = w2;
                break;
            }
        }

        return new double[]{w1, w2};
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package frenchsentimentclassification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Amine
 */
public class FrenchSentimentClassification {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        //String file=args[0];
        String file="test/task1-train.arff";
        
        String propPath="test/config.properties";
        Properties prop = new Properties();
	InputStream input = new FileInputStream(propPath);
        prop.load(input);
        
        int nbFolds=3;
        BufferedReader r;
        Instances data;
        
        ArrayList<Instances> trains = new ArrayList<Instances>();
        ArrayList<Instances> tests = new ArrayList<Instances>();
        
        r = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        data = new Instances(r);
        r.close();
        // Generate CV sets
        
        Random rand = new Random();   // create seeded number generator
        data.randomize(rand);        // randomize data with number generator
        data.setClass(data.attribute("_class"));
        data.stratify(nbFolds);
        for (int f=0; f<nbFolds; f++){
            trains.add(data.trainCV(nbFolds,f));
            tests.add(data.testCV(nbFolds,f));
        }
        
        /*SearchBestConfigurations sbc = new SearchBestConfigurations(trains,tests);
        
        // Find the best ngrams
        prop = sbc.bestNgrams(prop);
        // Estimate the best complexity parameter
        
        // Find the best syntatic
        
        // Find the best ngrams
*/


        String propPath2="test/configU.properties";
        Properties propC = new Properties();
	InputStream inputC = new FileInputStream(propPath2);
        propC.load(inputC);   
        inputC.close();
        
        Instances trai, tes;
        double mi=0;
        for (int i=0; i<trains.size(); i++){
            trai = trains.get(i);
            tes = tests.get(i);
            System.out.print(tes.numAttributes()+"\t");
            StringToWordVector filter = Tokenisation.WordNgrams(propC);
            filter.setInputFormat(trai);
            trai = Filter.useFilter(trai, filter);
            tes = Filter.useFilter(tes, filter);
            System.out.print(tes.numAttributes()+"\t");
            trai.setClass(trai.attribute("_class"));
            tes.setClass(trai.attribute("_class"));
            SMO classifier = new SMO();
            classifier.buildClassifier(trai);
            Evaluation eTest = new Evaluation(trai);
            eTest.evaluateModel(classifier, tes);
            mi += eTest.unweightedMicroFmeasure();
            System.out.println(tes.size()+" "+eTest.unweightedMicroFmeasure());
            //saveFile(test,"test/"+propB.getProperty("Ngrams.min")+propB.getProperty("Ngrams.max")+"-"+i+".arrff");
        }
        System.out.println(mi/trains.size());
        
        
        String propPath1="test/configB.properties";
        Properties propB = new Properties();
	InputStream inputB = new FileInputStream(propPath1);
        propB.load(inputB);
        inputB.close();
        
        Instances train, test;
        double miF=0;
        for (int i=0; i<trains.size(); i++){
            train = trains.get(i);
            test = tests.get(i);
            System.out.print(test.numAttributes()+"\t");
            StringToWordVector filter = Tokenisation.WordNgrams(propB);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            System.out.print(test.numAttributes()+"\t");
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            SMO classifier2 = new SMO();
            classifier2.buildClassifier(train);
            Evaluation eTest2 = new Evaluation(train);
            eTest2.evaluateModel(classifier2, test);
            miF += eTest2.unweightedMicroFmeasure();
            System.out.println(test.size()+" "+eTest2.unweightedMicroFmeasure());
            //saveFile(test,"test/"+propB.getProperty("Ngrams.min")+propB.getProperty("Ngrams.max")+"-"+i+".arrff");
        }
        System.out.println(miF/trains.size());
        
        
        
        
        
    }
    
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package frenchsentimentclassification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;
import weka.core.Instances;

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
        
        int nbFolds=10;
        BufferedReader r;
        Instances data;
        
        ArrayList<Instances> trains = new ArrayList<Instances>();
        ArrayList<Instances> tests = new ArrayList<Instances>();
        
        r = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        data = new Instances(r);
        
        // Generate CV sets
        
        Random rand = new Random();   // create seeded number generator
        data.randomize(rand);        // randomize data with number generator
        data.setClass(data.attribute("_class"));
        data.stratify(nbFolds);
        for (int f=0; f<nbFolds; f++){
            trains.add(data.trainCV(nbFolds,f));
            tests.add(data.testCV(nbFolds,f));
        }
        
        // Find the best ngrams
        prop = SearchBestConfigurations.bestNgrams(trains, tests, prop);
        // Estimate the best complexity parameter
        
        // Find the best ngrams
        
        // Find the best ngrams
        
        
        
    }
    
}

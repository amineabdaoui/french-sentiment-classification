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
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.tokenizers.NGramTokenizer;
import static weka.estimators.Estimator.clone;
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
        
        PrintWriter Out = new PrintWriter("SaveResults.txt");
        
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
        r.close();
        // Generate CV sets
        
        Random rand = new Random();   // create seeded number generator
        data.randomize(rand);        // randomize data with number generator
        data.setClass(data.attribute("_class"));
        data.stratify(nbFolds);
        for (int f=0; f<nbFolds; f++){
            Instances Tr = data.trainCV(nbFolds,f);
            Instances Te = data.testCV(nbFolds,f);
            trains.add(Tr);            
            tests.add(Te);
        }
        
        
        SearchBestConfigurations sbc = new SearchBestConfigurations(trains,tests);
        // Find the best ngrams
        System.out.println("####################");
        System.out.println("       Ngrams");
        System.out.println("####################");
        //prop = sbc.bestNgrams(prop);
        PropWithMeasure ngrams = sbc.bestNgrams(prop, Out);
        prop = ngrams.getProp();
        double measure = ngrams.getMeasure();
        // Estimate the best complexity parameter
        
        // Find the best syntatic
        
        // Find the preprocessing
        String [] Pretraitements = {"Preprocessings.normalizeHyperlinks","Preprocessings.normalizeSlang",
            "Preprocessings.removeStopWords","Preprocessings.lowercase","Preprocessings.normalizeEmails",
            "Preprocessings.replacePseudonyms","Preprocessings.lemmatize"};
        PropWithMeasure preproc = sbc.bestConfig(prop, measure, Pretraitements, Pretraitements.length, Out);
        prop = preproc.getProp();
        
        Out.close();
    }
    
}

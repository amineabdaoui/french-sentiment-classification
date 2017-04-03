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
        String file=args[0];
        //String file="test/task1-train.arff";
        
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
        System.out.println("#####################");
        System.out.println("       Ngrams");
        System.out.println("#####################");
        Out.println("#####################");
        Out.println("       Ngrams");
        Out.println("#####################");
        Out.flush();
        //prop = sbc.bestNgrams(prop);
        PropWithMeasure ngrams = sbc.bestNgrams(prop, Out);
        prop = ngrams.getProp();
        double measure = ngrams.getMeasure();
        
        // Evalute the feature selection
        System.out.println("#############################");
        System.out.println("       Feture Selection");
        System.out.println("#############################");
        Out.println("#############################");
        Out.println("       Feture Selection");
        Out.println("#############################");
        Out.flush();
        measure = sbc.AttributeSelection(prop, measure, Out);
        
        // Estimate the best complexity parameter
        System.out.println("#############################");
        System.out.println("       Complexity Parameter");
        System.out.println("#############################");
        Out.println("#############################");
        Out.println("       Complexity Parameter");
        Out.println("#############################");
        Out.flush();
        PropWithMeasure ComSVM = sbc.bestComSVM(prop, measure, Out);
        prop = ComSVM.getProp();
        measure = ComSVM.getMeasure();
        
        // Choose the best preprocessings
        System.out.println("############################");
        System.out.println("       Preprocessings");
        System.out.println("############################");
        Out.println("############################");
        Out.println("       Preprocessings");
        Out.println("############################");
        Out.flush();
        String [] Pretraitements = {"Preprocessings.normalizeHyperlinks","Preprocessings.normalizeSlang",
            "Preprocessings.removeStopWords","Preprocessings.lowercase","Preprocessings.normalizeEmails",
            "Preprocessings.replacePseudonyms","Preprocessings.lemmatize"};
        PropWithMeasure preproc = sbc.bestConfig(prop, measure, Pretraitements, Pretraitements.length, Out);
        prop = preproc.getProp();
        measure = preproc.getMeasure();
        Out.println("Old measure : "+measure+"\n");
        
        // Choose the best syntatic features
        System.out.println("##############################");
        System.out.println("      Lexicon features");
        System.out.println("##############################");
        Out.println("############################");
        Out.println("       Lexicon features");
        Out.println("############################");
        Out.flush();
        String [] LexiconFeatures = {"Lexicons.feelPol","Lexicons.affectsPol","Lexicons.dikoPol","Lexicons.polarimotsPol",
            "Lexicons.affectsEmo","Lexicons.dikoEmo","Lexicons.feelEmo"};
        PropWithMeasure Lexicon = sbc.bestConfig(prop, measure, LexiconFeatures, LexiconFeatures.length, Out);
        prop = Lexicon.getProp();
        
        // Choose the best syntatic features
        System.out.println("##############################");
        System.out.println("      Syntatic features");
        System.out.println("##############################");
        Out.println("############################");
        Out.println("       Syntatic features");
        Out.println("############################");
        Out.flush();
        String [] SyntacticFeatures = {"SyntacticFeatures.countHashtags","SyntacticFeatures.presencePunctuation",
            "SyntacticFeatures.countElongatedWords","SyntacticFeatures.countCapitalizations","SyntacticFeatures.countNegators",
            "SyntacticFeatures.presenceSmileys","SyntacticFeatures.presencePartOfSpeechTags"};
        PropWithMeasure Syntactic = sbc.bestConfig(prop, measure, SyntacticFeatures, SyntacticFeatures.length, Out);
        prop = Syntactic.getProp();
        
        // Evalute the feature selection
        System.out.println("#############################");
        System.out.println("       Feture Selection");
        System.out.println("#############################");
        Out.println("#############################");
        Out.println("       Feture Selection");
        Out.println("#############################");
        Out.flush();
        measure = sbc.AttributeSelection(prop, measure, Out);
        
        // Estimate the best complexity parameter
        System.out.println("#############################");
        System.out.println("       Complexity Parameter");
        System.out.println("#############################");
        Out.println("#############################");
        Out.println("       Complexity Parameter");
        Out.println("#############################");
        Out.flush();
        ComSVM = sbc.bestComSVM(prop, measure, Out);
        prop = ComSVM.getProp();
        measure = ComSVM.getMeasure();
        System.out.println("Best : "+measure);
        Out.println("Best : "+measure);
        
        
        Out.close();
    }
     
}

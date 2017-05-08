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
import weka.core.Instance;
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
        String fileTrain=args[0];
        String fileTest=args[1];
        
        PrintWriter Out = new PrintWriter("equipe-4_tache1_run.csv");
        
        String propPath="test/config.properties";
        Properties prop = new Properties();
	InputStream input = new FileInputStream(propPath);
        prop.load(input);
        /*
        int nbFolds=10;
        BufferedReader r;
        Instances data;
        
        ArrayList<Instances> trains = new ArrayList<Instances>();
        ArrayList<Instances> tests = new ArrayList<Instances>();
        
        r = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        data = new Instances(r);
        r.close();
        // Generate CV sets
        
        /*Random rand = new Random();   // create seeded number generator
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
        
        // Create Ngrams - specific
        sbc.setInstancesNgrams(prop);
        double measure = sbc.run(prop);
        System.out.println("Measure = "+measure);
        Out.println("Measure = "+measure);
        Out.flush();
        /*
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
        Out.println("Measure : "+measure+"\n");
        
        // Choose the best lexicon features
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
        measure = Lexicon.getMeasure();
        prop = Lexicon.getProp();
        
        // Choose the best incontiguity features
        System.out.println("##############################");
        System.out.println("      Incontiguity features");
        System.out.println("##############################");
        Out.println("############################");
        Out.println("       Incontiguity features");
        Out.println("############################");
        Out.flush();
        String [] IncontiguityFeatures = {"Lexicons.incongruity","Lexicons.incongruityAll"};
        PropWithMeasure Incontiguity = sbc.bestConfig(prop, measure, IncontiguityFeatures, IncontiguityFeatures.length, Out);
        measure = Incontiguity.getMeasure();
        prop = Incontiguity.getProp();
        
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
        measure = Syntactic.getMeasure();
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
        System.out.println("       Complexity Parameter 2");
        System.out.println("#############################");
        Out.println("#############################");
        Out.println("       Complexity Parameter 2");
        Out.println("#############################");
        Out.flush();
        
        StringToWordVector filter = Tokenisation.WordNgrams(prop);
        ConstructionARFF obj = new ConstructionARFF(prop);
        data = obj.ConstructionInstances(data);
        /*filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        data.setClass(data.attribute("_class"));
        for (double c=0.05;c<=5;c+=0.05){
            double moy=0;
            for (int i=0; i<10; i++){
                double miF=0;
                Random rand = new Random();   // create seeded number generator
                data.randomize(rand);        // randomize data with number generator
                data.setClass(data.attribute("_class"));
                data.stratify(nbFolds);
                for (int f=0; f<nbFolds; f++){
                    Instances Tr = data.trainCV(nbFolds,f);
                    Instances Te = data.testCV(nbFolds,f);
                    filter.setInputFormat(Tr);
                    Tr = Filter.useFilter(Tr, filter);
                    Te = Filter.useFilter(Te, filter);
                    Tr.setClass(Tr.attribute("_class"));
                    Te.setClass(Te.attribute("_class"));
                    SMO classifier = new SMO();
                    classifier.setC(c);
                    classifier.buildClassifier(Tr);
                    Evaluation eTest = new Evaluation(Tr);
                    eTest.evaluateModel(classifier, Te);
                    if (prop.getProperty("measure").equals("micro")) miF += eTest.unweightedMicroFmeasure();
                    else miF += eTest.unweightedMacroFmeasure();
                }
                miF=miF/nbFolds;
                System.out.println(miF);
                moy += miF;
            }
            System.out.println(c+";"+moy/10);
            Out.println(c+";"+moy/10);
            Out.flush();
        }
        
        Out.close();*/
        
        // Apply model
        
        System.out.println("Chargement des fichiers");
        BufferedReader rTrain = new BufferedReader(new InputStreamReader(new FileInputStream(fileTrain), "UTF-8"));
        BufferedReader rTest= new BufferedReader(new InputStreamReader(new FileInputStream(fileTest), "UTF-8"));
        
        Instances Train = new Instances(rTrain);
        Instances Test = new Instances(rTest);
        Instances TestS = new Instances(Test);
        
        System.out.println("Contruction ARFF");
        ConstructionARFF obj = new ConstructionARFF(prop);
        Train = obj.ConstructionInstances(Train);
        Test = obj.ConstructionInstances(Test);
        
        System.out.println("Tokenisation");
        StringToWordVector filter = Tokenisation.WordNgrams(prop);
        filter.setInputFormat(Train);
        Train = Filter.useFilter(Train, filter);
        Test = Filter.useFilter(Test, filter);
        Train.setClass(Train.attribute("_class"));
        Test.setClass(Test.attribute("_class"));
        
        double c=Double.parseDouble(prop.getProperty("SVM.CompexityParameter"));
        System.out.println("Aprentissage c="+c);
        SMO classifier = new SMO();
        classifier.setC(c);
        classifier.buildClassifier(Train);
        double classe;
        String classeText;
        System.out.println("Classification");
        for (int i=0; i<Test.size(); i++){
            classe = classifier.classifyInstance(Test.instance(i));
            if (classe==0) classeText="positive";
            else if (classe==1) classeText="negative";
            else if (classe==2) classeText="objective";
            else classeText="mixed";
            Out.println((i+1)+"\t"+TestS.instance(i).stringValue(TestS.attribute("_text"))+"\t"+classeText);
            Out.flush();
        }
        
    }
     
}

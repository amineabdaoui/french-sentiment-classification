package frenchsentimentclassification;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Amine
 */
public class SearchBestConfigurations {
    
    public static Properties bestNgrams(ArrayList<Instances> trains, ArrayList<Instances> tests, Properties prop) throws IOException, Exception{
        double miU=0, miB=0, miUB=0;
        System.out.println(prop.size());
        // evaluate unigrams
        FileOutputStream outU = new FileOutputStream("test/configU.properties");
        Properties propU = prop;
        propU.setProperty("Ngrams.min", "1");
        propU.setProperty("Ngrams.max", "1");
        propU.store(outU, null);
        outU.close();
        miU = runNgrams(trains,tests,prop);
        System.out.println("miU = "+miU);
        // evaluate bigrams
        FileOutputStream outB = new FileOutputStream("test/configB.properties");
        Properties propB = prop;
        propB.setProperty("Ngrams.min", "2");
        propB.setProperty("Ngrams.max", "2");
        propB.store(outB, null);
        outB.close();
        miB = runNgrams(trains,tests,propB);
        System.out.println("miB = "+miB);
        // evaluate unigrams+bigrams
        FileOutputStream outUB = new FileOutputStream("test/configUB.properties");
        Properties propUB = prop;
        propUB.setProperty("Ngrams.min", "1");
        propUB.setProperty("Ngrams.max", "2");
        propUB.store(outUB, null);
        outUB.close();
        miUB = runNgrams(trains,tests,propUB);
        System.out.println("miUB = "+miUB);
        
        FileOutputStream out;
        if (miU >= miB && miU >= miUB){
            out = new FileOutputStream("test/config.properties");
            prop.setProperty("Ngrams.min", "1");
            prop.setProperty("Ngrams.max", "1");
            prop.store(out, null);
            out.close();
        } else if (miB >= miU && miB >= miUB){
            out = new FileOutputStream("test/config.properties");
            prop.setProperty("Ngrams.min", "2");
            prop.setProperty("Ngrams.max", "2");
            prop.store(out, null);
            out.close();
        }
        
        return prop;
    }
    
    public static double runNgrams(ArrayList<Instances> trains, ArrayList<Instances> tests, Properties props) throws FileNotFoundException, IOException, Exception{
        Instances train, test;
        double miF=0;
        for (int i=0; i<trains.size(); i++){
            train = trains.get(i);
            test = tests.get(i);
            System.out.print(test.numAttributes()+"\t");
            StringToWordVector filter = Tokenisation.WordNgrams(props);
            filter.setInputFormat(train);
            train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);
            System.out.print(test.numAttributes()+"\t");
            train.setClass(train.attribute("_class"));
            test.setClass(train.attribute("_class"));
            SMO classifier = new SMO();
            classifier.buildClassifier(train);
            Evaluation eTest = new Evaluation(train);
            eTest.evaluateModel(classifier, test);
            miF += eTest.unweightedMicroFmeasure();
            System.out.println(test.size()+" "+eTest.unweightedMicroFmeasure());
        }
        return miF/trains.size();
    }
    
}

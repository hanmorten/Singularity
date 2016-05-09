package org.singularity.algorithms.ranking;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.classifier.*;
import org.singularity.algorithms.kernel.*;
import org.singularity.algorithms.regression.*;
import org.singularity.algorithms.neuralnet.*;
import org.singularity.algorithms.neuralnet.error.ErrorFunction;
import org.singularity.algorithms.neuralnet.input.InputFunction;
import org.singularity.algorithms.neuralnet.learning.LearningRule;
import org.singularity.algorithms.neuralnet.output.OutputFunction;
/**
 * Test class for ranking algorithms.
 */
public class RankingTest {

	private static class Travellers implements RankingSubject {
		
		private String name;
		
		private RealVector features;
		
		public Travellers(String name, double adults, double children, double dow) {
			this.name = name;
			final double[] features = new double[3];
			features[0] = adults;
			features[1] = children;
			features[2] = dow;
			this.features = new ArrayRealVector(features);
		}
		
		public RealVector getFeatures() {
			return this.features;
		}
		
		public String toString() {
			return name;
		}
		
	}
	
	private static class Hotel implements RankingObject {
		
		private String name;
		
		private RealVector features;
		
		public Hotel(String name, double starrating, double tripadvisor, double city, double resort, double price) {
			this.name = name;
			final double[] features = new double[5];
			features[0] = starrating;
			features[1] = tripadvisor;
			features[2] = city;
			features[3] = resort;
			features[4] = price;
			this.features = new ArrayRealVector(features);
		}
		
		public RealVector getFeatures() {
			return this.features;
		}

		public String toString() {
			return name;
		}

	}

	/**
	 * Create some sample traveling parties.
	 * Feature vector: Adults, Children, Day-of-Week (Monday is 1.0)
	 */
	private static Travellers business = new Travellers("Business",1,0,1);
	private static Travellers couple = new Travellers("Couple",2,0,0);
	private static Travellers family1 = new Travellers("Family1",2,2,0);
	private static Travellers family2 = new Travellers("Family2",2,3,0);
	
	private static List<RankingSubject> travellers = new ArrayList<RankingSubject>();
	static {
		travellers.add(business);
		travellers.add(couple);
		travellers.add(family1);
		travellers.add(family2);
	}
	
	/**
	 * Create some sample hotels.
	 * Feature vector: Star-rating, TripAdvisor rating, is-city, is-resort, price bracket.
	 */
	private static Hotel h5sc = new Hotel("5-star City",      2.0, 0.0, 2.0, 0.0, 0.0);
	private static Hotel h5sr = new Hotel("5-star Resort",    2.0, 0.0, 0.0, 2.0, 0.0);
	private static Hotel h5ss = new Hotel("5-star Cheap",     2.0, 0.0, 0.0, 0.0, 2.0);
	private static Hotel h4sc = new Hotel("4-star City",      1.0, 0.0, 2.0, 0.0, 0.0);
	private static Hotel h4sr = new Hotel("4-star Resort",    1.0, 0.0, 0.0, 2.0, 0.0);
	private static Hotel h4ss = new Hotel("4-star Cheap",     1.0, 0.0, 0.0, 0.0, 2.0);
	private static Hotel h3sc = new Hotel("3-star City",      0.0, 0.0, 2.0, 0.0, 0.0);
	private static Hotel h3sr = new Hotel("3-star Resort",    0.0, 0.0, 0.0, 2.0, 0.0);
	private static Hotel h3ss = new Hotel("3-star Cheap",     0.0, 0.0, 0.0, 0.0, 2.0);

	private static List<RankingObject> hotels = new ArrayList<RankingObject>();
	static {
		hotels.add(h5sc);
		hotels.add(h5sr);
		hotels.add(h5ss);
		hotels.add(h4sc);
		hotels.add(h4sr);
		hotels.add(h4ss);
		hotels.add(h3sc);
		hotels.add(h3sr);
		hotels.add(h3ss);
	}

	//private Ranking ranking = new PairwiseRanking(new Boosting(100));
	//private Ranking ranking = new PairwiseRanking(new SupportVectorMachine(new PolynomialKernel(1.0d, 0.1d, 2d)));
	//private Ranking ranking = new PairwiseRanking(new SupportVectorMachine(new LinearKernel()));
	private Ranking ranking = null;//new PointwiseRanking(new PoissonRegression(500));
	
	public RankingTest() {
		final NeuralNetworkConfiguration config = new NeuralNetworkConfiguration();
		config.setErrorFunction(ErrorFunction.Type.MEAN_SQUARED);
		config.setInputFunction(InputFunction.Type.WEIGHTED_SUM);
		config.setLearningRule(LearningRule.Type.BACK_PROPAGATION, 0.25);
		config.setOutputFunction(OutputFunction.Type.SIGMOID);
		config.setIterations(200);
		config.addLayer(13, true);
		config.addLayer(50, true);
		config.addLayer(1, false);
		final NeuralNetwork network = new NeuralNetwork(config);
		ranking = new PairwiseRanking(network);
	}

	public double accuracy() throws RankingException {
		return ranking.accuracy();
	}
	
	public void booked(Travellers travellers, Hotel hotel) {
		for (int i=0; i<hotels.size(); i++) {
			final Hotel other = (Hotel)hotels.get(i);
			if (other != hotel) this.ranking.addSample(travellers, other, hotel);
		}
	}

	public void rating(Travellers travellers, Hotel[] hotels) {
		for (int i=0; i<hotels.length-1; i++) {
			final Hotel hotel = hotels[i];
			final Hotel other = hotels[i+1];
			this.ranking.addSample(travellers, other, hotel);
		}
	}

	public void train() throws RankingException {
		this.ranking.train();
	}
	
	public void test(Travellers travellers) throws RankingException {
		System.out.println("Travellers: "+travellers);
		List<RankingObject> clone = new ArrayList<RankingObject>();
		clone.addAll(RankingTest.hotels);
		this.ranking.sort(travellers, clone);
		for (int j=0; j<clone.size(); j++) {
			final Hotel hotel = (Hotel)clone.get(j);
			System.out.println("  "+j+": "+hotel);
		}
	}

	public static void main(String[] args) {
		try {
			RankingTest tester = new RankingTest();
			tester.rating(business, new Hotel[] { h4sc, h4sc, h4sc, h3sc, h4ss, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.rating(business, new Hotel[] { h4sc, h3sc, h4ss, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.rating(business, new Hotel[] { h4sc, h4sc, h4sc, h3sc, h4ss, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.rating(business, new Hotel[] { h4ss, h4sc, h4sc, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.rating(business, new Hotel[] { h4sc, h4sc, h4sc, h3sc, h4ss, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.rating(business, new Hotel[] { h4ss, h4sr, h4ss, h4ss, h5ss, h3ss, h4sr, h5sr, h3sr });
			tester.train();
			System.err.println("Accuracy: "+tester.accuracy());
			tester.test(business);

			tester = new RankingTest();
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.rating(couple, new Hotel[] { h5sc, h5sc, h5ss, h4ss, h4ss, h3ss, h4sr, h3sr, h3sr });
			tester.train();
			System.err.println("Accuracy: "+tester.accuracy());
			tester.test(couple);
			
			tester = new RankingTest();
			tester.rating(family1, new Hotel[] { h4sr, h4ss, h3sr, h3ss, h4sc, h3sc, h5ss, h5sr, h5sc });
			tester.train();
			System.err.println("Accuracy: "+tester.accuracy());
			tester.test(family1);
			
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}
	}
	
}


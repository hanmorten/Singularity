package org.singularity.application.calendar;

import java.util.*;

import org.apache.commons.math3.linear.*;

/**
 * This class maps Java Calendar instances into a feature vector, so that we
 * can train a regression algorithm using simple calendar events.
 */
public class CalendarEvents {

	/**
	 * Base class for all calendar events. Extensions of this class are
	 * responsible for detecting if a specific event occurred on a given date.
	 */
	protected abstract class CalendarEvent {
		
		/**
		 * Detects if the event handled by the extending class occurred on a
		 * given date.
		 * @param date Date to check if event occurred on.
		 * @return true if event occurred on given date.
		 */
		protected abstract boolean match(Calendar date);
		
	}

	protected class DayOfWeek extends CalendarEvent {
		
		private int day = 0;
		
		public DayOfWeek(int day) {
			this.day = day;
		}
		
		protected boolean match(Calendar date) {
			final int day = date.get(Calendar.DAY_OF_WEEK);
			return day == this.day;
		}
		
	}

	protected class Month extends CalendarEvent {
		
		private int month = 0;
		
		public Month(int month) {
			this.month = month;
		}
		
		protected boolean match(Calendar date) {
			final int month = date.get(Calendar.MONTH);
			return month == this.month;
		}
		
	}
	
	protected class DayOfMonth extends CalendarEvent {
		
		private int day = 0;
		
		public DayOfMonth(int day) {
			this.day = day;
		}
		
		protected boolean match(Calendar date) {
			final int day = date.get(Calendar.DAY_OF_MONTH);
			return day == this.day;
		}
		
	}

	protected class NewYear extends CalendarEvent {
		
		protected boolean match(Calendar date) {
			final int month = date.get(Calendar.MONTH);
			final int day = date.get(Calendar.DAY_OF_MONTH);
			if (month == Calendar.JANUARY && day == 1) return true;
			if (month == Calendar.DECEMBER && day == 31) return true;
			return false;
		}
		
	}
	
	protected class Christmas extends CalendarEvent {
		
		protected boolean match(Calendar date) {
			final int month = date.get(Calendar.MONTH);
			if (month != Calendar.DECEMBER) return false;
			final int day = date.get(Calendar.DAY_OF_MONTH);
			return (day >= 24 && day <= 26);
		}
		
	}

	protected abstract class EasterBase extends CalendarEvent {
		
		public int getEasterSunday(Calendar date) {
			final int year = date.get(Calendar.YEAR);
			
			// This algorithm can detect Easter Sunday up until year 4096.
	    	final int a = year % 19;
	    	final int b = year / 100;
	    	final int c = year % 100;
	    	final int d = b / 4;
	    	final int e = b % 4;
	    	final int f = (b + 8) / 25;
	    	final int g = (b - f + 1) / 3;
	    	final int h = (19 * a + b - d - g + 15) % 30;
	    	final int i = c / 4;
	    	final int k = c % 4;
	    	final int l = (32 + 2 * e + 2 * i - h - k) % 7;
	    	final int m = (a + 11 * h + 22 * l) / 451;
	    	final int month = (h + l - 7 * m + 114) / 31;
	    	final int day = (h + l - 7 * m + 114) % 31 + 1;
	    	
	    	final Calendar cal = Calendar.getInstance();
			cal.set(year, month - 1, day,0,0,0);
			cal.set(Calendar.MILLISECOND,0);
			return cal.get(Calendar.DAY_OF_YEAR); 
		}
	}

	protected class Easter extends EasterBase {
		
		protected boolean match(Calendar date) {
			final int start = super.getEasterSunday(date) - 8;
			final int end = start + 9;
			
			final int day = date.get(Calendar.DAY_OF_YEAR);
			return (day >= start && day <= end);
		}
		
	}

	protected class Pentecost extends EasterBase {
		
		protected boolean match(Calendar date) {
			final int start = super.getEasterSunday(date) + 49;
			final int end = start + 2;
			
			final int day = date.get(Calendar.DAY_OF_YEAR);
			return (day >= start && day <= end);
		}
		
	}
	
	/** List of calendar events we will track. */
	private final List<CalendarEvent> events = new ArrayList<CalendarEvent>();

	/**
	 * Creates a new calendar events tracker.
	 */
	public CalendarEvents() {
		this.events.add(new DayOfWeek(Calendar.MONDAY));
		this.events.add(new DayOfWeek(Calendar.TUESDAY));
		this.events.add(new DayOfWeek(Calendar.WEDNESDAY));
		this.events.add(new DayOfWeek(Calendar.THURSDAY));
		this.events.add(new DayOfWeek(Calendar.FRIDAY));
		this.events.add(new DayOfWeek(Calendar.SATURDAY));
		this.events.add(new DayOfWeek(Calendar.SUNDAY));
		
		this.events.add(new Month(Calendar.JANUARY));
		this.events.add(new Month(Calendar.FEBRUARY));
		this.events.add(new Month(Calendar.MARCH));
		this.events.add(new Month(Calendar.APRIL));
		this.events.add(new Month(Calendar.MAY));
		this.events.add(new Month(Calendar.JUNE));
		this.events.add(new Month(Calendar.JULY));
		this.events.add(new Month(Calendar.AUGUST));
		this.events.add(new Month(Calendar.SEPTEMBER));
		this.events.add(new Month(Calendar.OCTOBER));
		this.events.add(new Month(Calendar.NOVEMBER));
		this.events.add(new Month(Calendar.DECEMBER));

		for (int i=0; i<31; i++) {
			this.events.add(new DayOfMonth(i));
		}
		
		this.events.add(new NewYear());
		this.events.add(new Christmas());
		this.events.add(new Easter());
		this.events.add(new Pentecost());
	}
	
	/**
	 * Converts a date to a feature vector of calendar events.
	 * @param date Date to construct feature vector for.
	 * @return feature vector for this date.
	 */
	public RealVector getFeatureVector(Calendar date) {
		final int size = this.events.size();
		final RealVector features = new ArrayRealVector(size);
		for (int i=0; i<size; i++) {
			features.setEntry(i, this.events.get(i).match(date) ? 1.0 : 0.0);
		}
		return features;
	}

}

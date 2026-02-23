import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  ScrollView,
  StatusBar,
} from 'react-native';
import FontAwesome from '@expo/vector-icons/FontAwesome';

import { getResult } from '@/api/client';
import { theme } from '@/constants/theme';

type GradeInfo = { grade: string; label: string };

function getGrade(score: number): GradeInfo {
  if (score >= 90) return { grade: 'A', label: 'ELITE FORM' };
  if (score >= 75) return { grade: 'B', label: 'SOLID FORM' };
  if (score >= 60) return { grade: 'C', label: 'DEVELOPING' };
  if (score >= 45) return { grade: 'D', label: 'NEEDS WORK' };
  return { grade: 'F', label: 'REBUILD' };
}

const JOINT_METRICS = [
  { label: 'ELBOW' },
  { label: 'SHOULDER' },
  { label: 'KNEE' },
  { label: 'HIP' },
];

export default function ResultsScreen() {
  const [score, setScore] = useState<number | null>(null);
  const [strengths, setStrengths] = useState<string[]>([]);
  const [weaknesses, setWeaknesses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [animScore, setAnimScore] = useState(0);

  const ringScale = useRef(new Animated.Value(0.75)).current;
  const ringOpacity = useRef(new Animated.Value(0)).current;
  const feedbackSlide = useRef(new Animated.Value(32)).current;
  const feedbackOpacity = useRef(new Animated.Value(0)).current;
  const scoreValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    getResult('mock')
      .then((res) => {
        setScore(res.score);
        const listener = scoreValue.addListener(({ value }) => {
          setAnimScore(Math.round(value));
        });
        Animated.timing(scoreValue, {
          toValue: res.score,
          duration: 1100,
          delay: 500,
          useNativeDriver: false,
        }).start(() => scoreValue.removeListener(listener));
      })
      .catch(() => setScore(null))
      .finally(() => {
        setLoading(false);
        Animated.parallel([
          Animated.spring(ringScale, { toValue: 1, tension: 65, friction: 8, useNativeDriver: true }),
          Animated.timing(ringOpacity, { toValue: 1, duration: 450, useNativeDriver: true }),
          Animated.timing(feedbackOpacity, { toValue: 1, duration: 450, delay: 250, useNativeDriver: true }),
          Animated.spring(feedbackSlide, { toValue: 0, tension: 70, friction: 10, delay: 250, useNativeDriver: true }),
        ]).start();
      });
  }, []);

  const gradeInfo = score !== null ? getGrade(score) : null;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={theme.bg} />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.eyebrow}>ANALYSIS</Text>
        <Text style={styles.title}>RESULTS</Text>
        <View style={styles.accentBar} />
      </View>

      {loading ? (
        <View style={styles.loadingWrap}>
          <Text style={styles.loadingText}>ANALYZING...</Text>
        </View>
      ) : (
        <>
          {/* Score ring */}
          <Animated.View
            style={[
              styles.scoreSection,
              { opacity: ringOpacity, transform: [{ scale: ringScale }] },
            ]}>
            <View style={styles.scoreRing}>
              <Text style={styles.scoreNumber}>{animScore}</Text>
              <Text style={styles.scoreOutOf}>/ 100</Text>
            </View>

            {gradeInfo && (
              <View style={styles.gradeRow}>
                <Text style={styles.gradeText}>{gradeInfo.grade}</Text>
                <View style={styles.gradeDivider} />
                <Text style={styles.gradeLabel}>{gradeInfo.label}</Text>
              </View>
            )}
          </Animated.View>

          {/* Feedback cards */}
          <Animated.View
            style={{
              opacity: feedbackOpacity,
              transform: [{ translateY: feedbackSlide }],
            }}>

            {/* Strengths */}
            {strengths.length > 0 && (
              <View style={[styles.feedbackCard, styles.strengthCard]}>
                <View style={styles.feedbackHeader}>
                  <View style={[styles.feedbackDot, { backgroundColor: theme.success }]} />
                  <Text style={[styles.feedbackTitle, { color: theme.success }]}>STRENGTHS</Text>
                </View>
                {strengths.map((s, i) => (
                  <View key={i} style={styles.feedbackItem}>
                    <FontAwesome name="check" size={11} color={theme.success} />
                    <Text style={styles.feedbackItemText}>{s}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Weaknesses */}
            {weaknesses.length > 0 && (
              <View style={[styles.feedbackCard, styles.weaknessCard]}>
                <View style={styles.feedbackHeader}>
                  <View style={[styles.feedbackDot, { backgroundColor: theme.error }]} />
                  <Text style={[styles.feedbackTitle, { color: theme.error }]}>IMPROVE</Text>
                </View>
                {weaknesses.map((w, i) => (
                  <View key={i} style={styles.feedbackItem}>
                    <FontAwesome name="times" size={11} color={theme.error} />
                    <Text style={styles.feedbackItemText}>{w}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Placeholder when no feedback yet */}
            {strengths.length === 0 && weaknesses.length === 0 && (
              <View style={styles.placeholderCard}>
                <Text style={styles.placeholderIcon}>📊</Text>
                <Text style={styles.placeholderTitle}>DETAILED FEEDBACK</Text>
                <Text style={styles.placeholderSub}>
                  Strengths and weaknesses will appear here after a full analysis run.
                </Text>
              </View>
            )}

            {/* Joint angle metrics grid */}
            <View style={styles.metricsGrid}>
              {JOINT_METRICS.map(({ label }, i) => (
                <View
                  key={label}
                  style={[
                    styles.metricCell,
                    i % 2 === 0 && styles.metricCellRight,
                    i < 2 && styles.metricCellBottom,
                  ]}>
                  <Text style={styles.metricValue}>—°</Text>
                  <Text style={styles.metricLabel}>{label}</Text>
                </View>
              ))}
            </View>
          </Animated.View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: theme.bg,
  },
  container: {
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
  },
  header: {
    marginBottom: 36,
  },
  eyebrow: {
    fontSize: 11,
    letterSpacing: 5,
    color: theme.accent,
    fontFamily: 'SpaceMono',
    marginBottom: 6,
  },
  title: {
    fontSize: 40,
    fontWeight: '900',
    color: theme.textPrimary,
    letterSpacing: 2,
  },
  accentBar: {
    width: 40,
    height: 3,
    backgroundColor: theme.accent,
    marginTop: 12,
    borderRadius: 2,
  },
  loadingWrap: {
    paddingTop: 80,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 11,
    letterSpacing: 5,
    color: theme.textSecondary,
    fontFamily: 'SpaceMono',
  },
  scoreSection: {
    alignItems: 'center',
    marginBottom: 40,
  },
  scoreRing: {
    width: 180,
    height: 180,
    borderRadius: 90,
    borderWidth: 5,
    borderColor: theme.accent,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.surface,
    shadowColor: theme.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.35,
    shadowRadius: 24,
    elevation: 12,
    marginBottom: 20,
  },
  scoreNumber: {
    fontSize: 72,
    fontWeight: '900',
    color: theme.accent,
    lineHeight: 80,
    fontFamily: 'SpaceMono',
  },
  scoreOutOf: {
    fontSize: 13,
    color: theme.textSecondary,
    fontFamily: 'SpaceMono',
  },
  gradeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  gradeText: {
    fontSize: 36,
    fontWeight: '900',
    color: theme.textPrimary,
    letterSpacing: 2,
  },
  gradeDivider: {
    width: 1,
    height: 28,
    backgroundColor: theme.border,
  },
  gradeLabel: {
    fontSize: 11,
    letterSpacing: 4,
    color: theme.textSecondary,
    fontFamily: 'SpaceMono',
  },
  feedbackCard: {
    borderRadius: 4,
    borderWidth: 1,
    padding: 20,
    marginBottom: 12,
  },
  strengthCard: {
    borderColor: `${theme.success}44`,
    backgroundColor: theme.successDim,
  },
  weaknessCard: {
    borderColor: `${theme.error}44`,
    backgroundColor: theme.errorDim,
  },
  feedbackHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 14,
  },
  feedbackDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  feedbackTitle: {
    fontSize: 10,
    letterSpacing: 4,
    fontWeight: '700',
  },
  feedbackItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 8,
  },
  feedbackItemText: {
    fontSize: 14,
    color: theme.textPrimary,
    flex: 1,
    lineHeight: 20,
    marginTop: -1,
  },
  placeholderCard: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    padding: 32,
    alignItems: 'center',
    marginBottom: 12,
  },
  placeholderIcon: {
    fontSize: 36,
    marginBottom: 14,
  },
  placeholderTitle: {
    fontSize: 10,
    letterSpacing: 4,
    color: theme.textSecondary,
    marginBottom: 10,
  },
  placeholderSub: {
    fontSize: 13,
    color: theme.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
  },
  metricsGrid: {
    backgroundColor: theme.surface,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 4,
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  metricCell: {
    width: '50%',
    padding: 18,
    alignItems: 'center',
  },
  metricCellRight: {
    borderRightWidth: 1,
    borderRightColor: theme.border,
  },
  metricCellBottom: {
    borderBottomWidth: 1,
    borderBottomColor: theme.border,
  },
  metricValue: {
    fontSize: 22,
    fontWeight: '800',
    color: theme.textPrimary,
    fontFamily: 'SpaceMono',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 9,
    letterSpacing: 3,
    color: theme.textSecondary,
  },
});

def show_all_matches(regexes, subject, re_length=6):
    print('Sentence:')
    print()
    print('    {}'.format(subject))
    print()
    print(' regexp{} | matches'.format(' ' * (re_length - 6)))
    print(' ------{} | -------'.format(' ' * (re_length - 6)))
    for regexp in regexes:
        fmt = ' {:<%d} | {!r}' % re_length
        matches = re.findall(regexp, subject)
        if len(matches) > 8:
            matches = matches[:8] + ['...']
        print(fmt.format(regexp, matches))